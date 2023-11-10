import torch
import numpy as np
import cv2
import queue

def postprocess(filename, original, text_predictions, kernel_predictions, similarity_predictions, threshold:float = 0.5, minimum_area:float = 10):
    """
    1. text_predictions: (N, 1, H, W)
    2. kernel_predictions: (N, 1, H, W)
    3. similarity_predictions: (N, SC, H, W) - SC: similarity channels
    3. threshold: pixel-level threshold
    4. minimum_area: min area to filter out some "noise-like" predictions
    """
    
    batch_size = text_predictions.shape[0]
    
    text_predictions = torch.sigmoid(text_predictions).detach().cpu().numpy().astype(np.float32)
    kernel_predictions = torch.sigmoid(kernel_predictions).detach().cpu().numpy().astype(np.float32)
    similarity_predictions = similarity_predictions.detach().cpu().numpy().astype(np.float32)
    
    text_regions = text_predictions > threshold
    kernel_regions = (kernel_predictions > threshold) * text_regions
    
    
    all_reconstruct = []
    for i in range(batch_size):
        reconstuct_item = reconstruct_single(text_region=text_regions[i], kernel_region=kernel_regions[i], similarity_vector=similarity_predictions[i]
                                             ,minimum_area=minimum_area, distance_threhold=10)
        reconstuct_item = reconstuct_item[np.newaxis, :]
        all_reconstruct.append((filename, original, reconstuct_item))
        
    return all_reconstruct
    

def reconstruct_single(text_region, kernel_region, similarity_vector, minimum_area, distance_threhold):
    
    text_region = text_region.squeeze().squeeze().astype(np.uint8)
    kernel_region = kernel_region.squeeze().squeeze().astype(np.uint8)
    
    num_kernels, kernel_map = cv2.connectedComponents(kernel_region, connectivity=4)
    
    kernel_map, valid_labels = filter_region_areas(num_regions=num_kernels, region_map=kernel_map, minimum_area=minimum_area)
    
    final_prediction = np.zeros_like(kernel_map, dtype=np.uint8)
    
    # Comparing pixels
    K_pixel_coordinates = np.argwhere(kernel_map > 0)
    
    K_pixel_coordinates_with_label = []
    
    
    
    # First, assign value of kernel to final_prediction
    for pixel in K_pixel_coordinates:
        x = pixel[1]
        y = pixel[0]
        label_value = kernel_map[y, x]
        K_pixel_coordinates_with_label.append(([y, x], label_value))
        final_prediction[y, x] = kernel_map[y, x]
    
    K_pixel_coordinates_with_label_queue = queue.Queue()
    for item in K_pixel_coordinates_with_label:
        K_pixel_coordinates_with_label_queue.put(item)
    
    kernel_sv = {}
    for i in valid_labels:
        kernel_mask = (kernel_map == i)[np.newaxis, :]
        kernel_mask = np.repeat(kernel_mask, repeats=4, axis=0)
        kernel_similarity_vector_value = similarity_vector[kernel_mask].mean(0)
        kernel_sv[i] = kernel_similarity_vector_value

    # 4-way finder
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    text_region_temp = text_region.copy()
    while not K_pixel_coordinates_with_label_queue.empty():
        item = K_pixel_coordinates_with_label_queue.get()
        # item: ([x, y], label)
        text_x = item[0][1]        
        text_y = item[0][0]
        current_label_value = item[1]
        current_kernel_sv = kernel_sv[current_label_value]
        
        # 4-way search
        for j in range(4):
            tmpx = text_x + dx[j]
            tmpy = text_y + dy[j]
            
            # Skip case where x, y is out of range
            if tmpx < 0 or tmpy >= text_region_temp.shape[0] or tmpy < 0 or tmpx >= text_region_temp.shape[1]:
                continue
            
            # Skip case where text_prediction = 0 but at kernel map is not.
            if text_region_temp[tmpy, tmpx] == 0 or final_prediction[tmpy, tmpx] > 0:
                continue
            
            dis = np.linalg.norm(similarity_vector[:, tmpy, tmpx] - current_kernel_sv)
            if dis >= distance_threhold:
                continue

            K_pixel_coordinates_with_label_queue.put(([tmpy, tmpx], current_label_value))
            final_prediction[tmpy, tmpx] = label_value
    
    return final_prediction
            
    
    
    
    
        
    
    
def filter_region_areas(num_regions, region_map, minimum_area):
    """
    Filter region that has area that smaller than minimum_area
    """
    filter_region_map = np.zeros_like(region_map, dtype=np.uint8)
    valid_labels = []
    
    if num_regions <= 1:
        return filter_region_map, valid_labels
    
    for region_idx in range(1, num_regions + 1):
        mask = (region_map == region_idx)
        region_sum = region_map[mask].sum()
        if region_sum < minimum_area:
            filter_region_map[mask] = 0
            continue
        
        filter_region_map[mask] = region_idx
        valid_labels.append(region_idx)
        
    return filter_region_map, valid_labels
    
    
    
    