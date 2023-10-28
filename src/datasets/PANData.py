import torch
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy import ndimage
from torch.utils.data import Dataset
from ..utils.common import calculating_scaling_offset, offset_polygon

class PANDataset(Dataset):
    """Some Information about PANDataset"""
    def __init__(self, images_folder, labels_folder, target_image_size, kernel_shrink_ratio):
        super(PANDataset, self).__init__()
        self.all_images_files = os.listdir(images_folder)
        
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.target_image_size = target_image_size
        
        self.kernel_shrink_ratio = kernel_shrink_ratio
        self.kekw = 0
        
    def __getitem__(self, index):
        filename = self.all_images_files[index]
        image_path = os.path.join(self.images_folder, filename)
        label_path = os.path.join(self.labels_folder, filename.replace(".jpg", ".xml"))
        return self.make_text_mask_and_kernel_mask(image_path=image_path, label_path=label_path, target_size=self.target_image_size)
        

    def __len__(self):
        return len(self.all_images_files)
    
    def extract_coords(self, segs):
        # Split the coordinates into pairs of X, Y values
        coords = [int(coord) for coord in segs.split(',')]
        return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    
    def make_text_mask_and_kernel_mask(self, image_path, label_path, target_size):
        image = cv2.imread(image_path)
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
        
        
        for box in root.findall(".//box"):
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
                
                cv2.fillPoly(text_mask, [pts], color=(1), lineType=cv2.LINE_AA)
                cv2.fillPoly(kernel_mask, [kernel_pts], color=(1), lineType=cv2.LINE_AA)
            except:
                # print(f"Found mis-label polygon in the file: {image_path}. Ignoring it!")
                continue
        
        cv2.imwrite(f"/home/huan/prjdir/panet-text-detection/data/debug_text_regions/text_mask_{self.kekw}.png", text_mask * 255)
        
        cv2.imwrite(f"/home/huan/prjdir/panet-text-detection/data/debug_kernel_regions/kernel_mask_{self.kekw}.png", kernel_mask * 255)
        
        self.kekw += 1
        
        text_mask_ndi_labels, _ = ndimage.label(text_mask)
        kernel_mask_ndi_labels, _ = ndimage.label(kernel_mask)
        
        return dict(image=torch.from_numpy(image.transpose((2, 0, 1))).float(), 
                    text_mask=torch.from_numpy(text_mask[np.newaxis, :]).float(), 
                    kernel_mask=torch.from_numpy(kernel_mask[np.newaxis, :]).float(),
                    text_mask_ndi_labels=torch.from_numpy(text_mask_ndi_labels).float().unsqueeze(0),
                    kernel_mask_ndi_labels=torch.from_numpy(kernel_mask_ndi_labels).float().unsqueeze(0))