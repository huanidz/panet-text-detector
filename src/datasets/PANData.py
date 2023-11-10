import torch
import os
import cv2
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug import augmenters as iaa
import imgaug as ia
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from ..utils.common import calculating_scaling_offset, offset_polygon
from imgaug import parameters as iap

class PANDataset(Dataset):
    """Some Information about PANDataset"""
    def __init__(self, images_folder, labels_folder, target_image_size, kernel_shrink_ratio, mode='train'):
        super(PANDataset, self).__init__()
        self.all_images_files = os.listdir(images_folder)
        
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.target_image_size = target_image_size
        
        self.kernel_shrink_ratio = kernel_shrink_ratio
        self.kekw = 0
        
        self.mode = mode
        
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        
        
    def __getitem__(self, index):
        if self.mode == 'train':
            filename = self.all_images_files[index]
            image_path = os.path.join(self.images_folder, filename)
            label_path = os.path.join(self.labels_folder, filename.replace(".jpg", ".xml"))
            return self.make_text_mask_and_kernel_mask(image_path=image_path, label_path=label_path, target_size=self.target_image_size)
        elif self.mode == 'eval':
            filename = self.all_images_files[index]
            image_path = os.path.join(self.images_folder, filename)
            image = cv2.imread(image_path)
            original = cv2.imread(image_path)
            image = cv2.resize(image, dsize=(self.target_image_size, self.target_image_size))
            img_mean = [0.485, 0.456, 0.406]
            img_std = [0.229, 0.224, 0.225]
            image = image / 255.0
            image = (image - img_mean) / img_std
            return original, torch.from_numpy(image.transpose((2, 0, 1))).float()
        elif self.mode == 'infer':
            filename = self.all_images_files[index]
            image_path = os.path.join(self.images_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            original = cv2.imread(image_path)
            image = cv2.resize(image, dsize=(self.target_image_size, self.target_image_size))
            img_mean = [0.485, 0.456, 0.406]
            img_std = [0.229, 0.224, 0.225]
            image = image / 255.0
            image = (image - img_mean) / img_std
            return filename, original, torch.from_numpy(image.transpose((2, 0, 1))).float()

    def __len__(self):
        return len(self.all_images_files)
    
    def extract_coords(self, segs):
        # Split the coordinates into pairs of X, Y values
        coords = [int(coord) for coord in segs.split(',')]
        return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    
    def make_text_mask_and_kernel_mask(self, image_path, label_path, target_size):
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        H, W, _ = image.shape
        scale_h = (target_size // 4) / H
        scale_w = (target_size // 4) / W
        
        image = cv2.resize(image, dsize=(target_size, target_size))        
        
        model_out_size = target_size // 4
        text_mask = np.zeros((model_out_size, model_out_size), dtype=np.int32)
        kernel_mask = np.zeros((model_out_size, model_out_size), dtype=np.int32)
        
        # Parse the XML label file
        tree = ET.parse(label_path)
        root = tree.getroot()
        
        all_boxes = root.findall(".//box")
        for idx, box in enumerate(all_boxes):
            segs = box.find("segs").text
            coords = self.extract_coords(segs)

            # Draw the polygon on the image
            pts = [list(coord) for coord in coords]
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            
            pts = [[coord[0] * scale_w, coord[1] * scale_h] for coord in coords]
            offset = calculating_scaling_offset(poly_coordinate=pts, r=self.kernel_shrink_ratio)
            kernel_pts = offset_polygon(polygon=pts, offset=offset)
            
            if len(kernel_pts) == 0 or len(pts) == 0:
                continue 
            
            try:
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                kernel_pts = np.array(kernel_pts, np.int32).reshape((-1, 1, 2))
                
                # NOTE: carefully, this can overlapping each others, cause the mask value to be 0 in the next processing step. This happen when use input image size is too small. Start to happen alot at 320x320.
                # NOTE: To prevent this happening, we will ignore the region that being overlap in the training process.
                cv2.fillPoly(text_mask, [pts], color=(idx + 1), lineType=cv2.LINE_AA)
                cv2.fillPoly(kernel_mask, [kernel_pts], color=(idx + 1), lineType=cv2.LINE_AA)
            except:
                # print(f"Found mis-label polygon in the file: {image_path}. Ignoring it!")
                continue
        
        # cv2.imwrite(f"/home/huan/prjdir/panet-text-detection/data/debug_text_regions/text_mask_{self.kekw}.png", text_mask * 255)
        
        # cv2.imwrite(f"/home/huan/prjdir/panet-text-detection/data/debug_kernel_regions/kernel_mask_{self.kekw}.png", kernel_mask * 255)
        
        # self.kekw += 1
        
        # text_mask_ndi_labels, _ = ndimage.label(text_mask)
        # kernel_mask_ndi_labels, _ = ndimage.label(kernel_mask)
        
        # aug_seed = np.random.randint(1, 1000)
        
        rotations = [0,90,180,270]
        
        augmenter = iaa.Sequential([
            iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}),
            iaa.Fliplr(p=0.5),
            iaa.Affine(rotate=iap.DeterministicList(rotations))
        ])
        
        augmenter = augmenter.to_deterministic()
        segmap_text = SegmentationMapsOnImage(text_mask, shape=text_mask.shape)
        segmap_kernel = SegmentationMapsOnImage(kernel_mask, shape=kernel_mask.shape)
        
        # Transform image
        transformed_image = augmenter.augment_image(image)
        # Transform segmentation maps separately
        transformed_segmap_text = augmenter.augment_segmentation_maps(segmap_text)
        transformed_segmap_kernel = augmenter.augment_segmentation_maps(segmap_kernel)
        
        image = transformed_image.copy()
        # cv2.imwrite("/home/huan/prjdir/panet-text-detection/album/image.png", image)
        # cv2.imwrite("/home/huan/prjdir/panet-text-detection/album/transformed_text_mask.png",transformed_segmap_text.arr * 15)
        # cv2.imwrite("/home/huan/prjdir/panet-text-detection/album/transformed_kernel_mask.png",transformed_segmap_kernel.arr * 15)
        
        # unique_values_text = np.unique(transformed_segmap_text.arr)
        # print("num of segtext:", unique_values_text)
        # unique_values_kernel = np.unique(transformed_segmap_kernel.arr)
        # print("num of kernel:", unique_values_kernel)
        
        
        # raise ValueError
        
        # raise ValueError("ASD")
        
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        image = image / 255.0
        image = (image - img_mean) / img_std
        
        text_mask_ndi_labels = transformed_segmap_text.arr.squeeze().copy()
        kernel_mask_ndi_labels = transformed_segmap_kernel.arr.squeeze().copy()
        
        return dict(image=torch.from_numpy(image.transpose((2, 0, 1))).float(), 
                    text_mask=torch.from_numpy(text_mask[np.newaxis, :]).float(), 
                    kernel_mask=torch.from_numpy(kernel_mask[np.newaxis, :]).float(),
                    text_mask_ndi_labels=torch.from_numpy(text_mask_ndi_labels).float().unsqueeze(0),
                    kernel_mask_ndi_labels=torch.from_numpy(kernel_mask_ndi_labels).float().unsqueeze(0))