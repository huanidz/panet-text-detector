import argparse
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from src.utils.common import offset_polygon, calculating_scaling_offset

def extract_coords(segs):
    # Split the coordinates into pairs of X, Y values
    coords = [int(coord) for coord in segs.split(',')]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

def visualize_annotations(image_folder, label_folder, output_folder):
    all_files = os.listdir(image_folder)
    for filename in tqdm(all_files):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace(".jpg", ".xml"))

            if not os.path.exists(label_path):
                continue

            # Read the image
            image = cv2.imread(image_path)

            # Parse the XML label file
            tree = ET.parse(label_path)
            root = tree.getroot()

            for box in root.findall(".//box"):
                segs = box.find("segs").text
                coords = extract_coords(segs)

                # Draw the polygon on the image
                pts = [list(coord) for coord in coords]
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                

            # Save the annotated image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

def make_mask(image_folder, label_folder, output_text_mask_folder, output_kernel_mask_folder, target_size:int=640, r:float=0.7, is_kernel:bool=False):
    all_files = os.listdir(image_folder)
    print("Start making text_mask and kernel_mask")
    for filename in tqdm(all_files):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace(".jpg", ".xml"))

            if not os.path.exists(label_path):
                continue
            
            # Read the image
            image = cv2.imread(image_path)
            H, W, C = image.shape
            scale_h = target_size / H
            scale_w = target_size / W
            
            image = cv2.resize(image, dsize=(target_size, target_size))
            text_mask = np.zeros(image.shape[0:2], dtype=np.int32)
            kernel_mask = np.zeros(image.shape[0:2], dtype=np.int32)
            
            # Parse the XML label file
            tree = ET.parse(label_path)
            root = tree.getroot()

            for box in root.findall(".//box"):
                segs = box.find("segs").text
                coords = extract_coords(segs)

                # Draw the polygon on the image
                pts = [[coord[0] * scale_w, coord[1] * scale_h] for coord in coords]
                offset = calculating_scaling_offset(poly_coordinate=pts, r=r)
                kernel_pts = offset_polygon(polygon=pts, offset=offset)
                
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                try:
                    kernel_pts = np.array(kernel_pts, np.int32).reshape((-1, 1, 2))
                except:
                    print(f"Found mis-label polygon in the file {filename}. Ignoring it!")
                    continue
                cv2.fillPoly(text_mask, [pts], color=(255), lineType=cv2.LINE_AA)
                cv2.fillPoly(kernel_mask, [kernel_pts], color=(255), lineType=cv2.LINE_AA)
                
            # Save the annotated image
            output_text_path = os.path.join(output_text_mask_folder, filename)
            output_kernel_path = os.path.join(output_kernel_mask_folder, filename)
            cv2.imwrite(output_text_path, text_mask)
            cv2.imwrite(output_kernel_path, kernel_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize text annotations from XML files on images")
    parser.add_argument("image_folder", help="Path to the folder containing images")
    parser.add_argument("label_folder", help="Path to the folder containing XML labels")
    # parser.add_argument("output_folder", help="Path to the output folder for visualizations")
    parser.add_argument("output_text_mask")
    parser.add_argument("output_kernel_mask")
    args = parser.parse_args()

    # if not os.path.exists(args.output_folder):
    #     os.makedirs(args.output_folder)
        
    if not os.path.exists(args.output_text_mask):
        os.makedirs(args.output_text_mask)
        
    if not os.path.exists(args.output_kernel_mask):
        os.makedirs(args.output_kernel_mask)

    # visualize_annotations(args.image_folder, args.label_folder, args.output_folder)
    make_mask(args.image_folder, args.label_folder, args.output_text_mask, args.output_kernel_mask)