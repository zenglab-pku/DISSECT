"""
Copyright Zexian Zeng's lab, AAIS, Peking Universit. All Rights Reserved

@author: Yufeng He
"""

import numpy as np
from numba import njit
import scipy
import cv2
from scipy import ndimage as ndi
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt
from skimage import draw
from skimage import filters, measure, morphology
from skimage.filters import threshold_multiotsu
from skimage.morphology import convex_hull_image
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.geometry import Polygon
from shapely.ops import unary_union
from joblib import Parallel, delayed

def create_kernel(kernel_size):
    """
    Create a kernel for convolution operation.

    Parameters
    ----------
    size : int
        The size of the square kernel.

    Returns
    -------
    kernel : ndarray
        The convolutional kernel.

    """
    # Check if the size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Size must be an odd number.")

    # Calculate the coordinates for the center of the kernel
    center = kernel_size // 2

    # Generate a grid of Euclidean distances from the center
    y, x = np.ogrid[: kernel_size, : kernel_size]
    distances = (x - center) ** 2 + (y - center) ** 2

    # Avoid division by zero at the center of the kernel
    distances[center, center] = 1

    # Create the kernel as the reciprocal of squared distances
    kernel = 1.0 / distances

    # Reset the center of the kernel to zero
    kernel[center, center] = 0

    return kernel

@njit()
def shift_image(matrix):
    """
    Shift an image in eight directions (up, down, left, right and the four diagonals).

    Parameters
    ----------
    img : ndarray
        The input image.

    Returns
    -------
    img_shifted : ndarray
        The image shifted in eight directions. The third dimension of the output corresponds to the eight directions.
    """
    height, width = matrix.shape
    img_shifted = np.zeros((height, width, 8))
    # Shift left-up
    img_shifted[1:, 1:, 0] = matrix[1:, 1:] - matrix[:-1, :-1]
    # Shift up
    img_shifted[1:, :, 1] = matrix[1:, :] - matrix[:-1, :]
    # Shift right-up
    img_shifted[1:, :-1, 2] = matrix[1:, :-1] - matrix[:-1, 1:]
    # Shift right
    img_shifted[:, :-1, 3] = matrix[:, :-1] - matrix[:, 1:]
    # Shift right-down
    img_shifted[:-1, :-1, 4] = matrix[:-1, :-1] - matrix[1:, 1:]
    # Shift down
    img_shifted[:-1, :, 5] = matrix[:-1, :] - matrix[1:, :]
    # Shift left-down
    img_shifted[:-1, 1:, 6] = matrix[:-1, 1:] - matrix[1:, :-1]
    # Shift left
    img_shifted[:, 1:, 7] = matrix[:, 1:] - matrix[:, :-1]
    
    return np.clip(-img_shifted, a_min=None, a_max=0)

def compute_gradient_map(img_cell, cell_box, gene_mtx, alpha=0.5, expand_by=2, gene=True):
    
    #776 ns +- 10.7 ns ; 782 +- 4.86 ; 761 +- 14.8 ns
    min_x, min_y, max_x, max_y = cell_box

    min_y = max(0, int(min_y - expand_by))
    max_y = min(img_cell.shape[0], int(max_y + expand_by))
    min_x = max(0, int(min_x - expand_by))
    max_x = min(img_cell.shape[1], int(max_x + expand_by))

    img_slice = img_cell[min_y : max_y + 1, min_x : max_x + 1]
    new_box = [min_x, min_y, max_x, max_y]
    
    # 134 µs ± 69.4 µs ; 106 µs ± 1.7 µs ; 106 µs ± 1.33 µs
    gradient_map = shift_image(img_slice)
    
    if gene:
        mask = (
            (gene_mtx[:, 1] >= min_x) & (gene_mtx[:, 1] <= max_x) &
            (gene_mtx[:, 2] >= min_y) & (gene_mtx[:, 2] <= max_y)
        )
        gene_sparse = gene_mtx[mask]
        gene_sparse[:, 1] -= min_x
        gene_sparse[:, 2] -= min_y

        length, width = img_slice.shape
        gene_order = np.unique(gene_sparse[:, 0])
        gene_to_index = {gene: idx for idx, gene in enumerate(list(gene_order))}
        gene_number = len(gene_order)
        gene_dense = np.zeros((length, width, gene_number), dtype=np.int32)
        
        if gene_number != 0:
            x = np.array(gene_sparse[:, 1], dtype=np.int32)
            y = np.array(gene_sparse[:, 2], dtype=np.int32)
            counts = gene_sparse[:, 3]
            gene_idx = gene_sparse[:, 0]
            try:
                gene_dense[y, x, np.array([gene_to_index[i] for i in gene_idx], dtype=np.int32)] = counts
            except:
                print(gene_sparse, x ,y, [gene_to_index[i] for i in gene_idx], gene_idx, gene_dense.shape)
        
        return img_slice, gradient_map, gene_dense, new_box
    
    return img_slice, gradient_map, None, new_box

@njit()
def shift_and_weight(gene_matrix, gene_weights, direction):
    matrix_shifted = gene_matrix.copy()
    gene_weights_shifted = gene_weights.copy()

    # Shift the matrix and gene weights in the specified direction
    if direction == 0:  # left-up
        matrix_shifted[1:, 1:, :] = gene_matrix[:-1, :-1, :]
        gene_weights_shifted[1:, 1:, :] = gene_weights[:-1, :-1, :]
    elif direction == 1:  # up
        matrix_shifted[1:, :, :] = gene_matrix[:-1, :, :]
        gene_weights_shifted[1:, :, :] = gene_weights[:-1, :, :]
    elif direction == 2:  # right-up
        matrix_shifted[1:, :-1, :] = gene_matrix[:-1, 1:, :]
        gene_weights_shifted[1:, :-1, :] = gene_weights[:-1, 1:, :]
    elif direction == 3:  # right
        matrix_shifted[:, :-1, :] = gene_matrix[:, 1:, :]
        gene_weights_shifted[:, :-1, :] = gene_weights[:, 1:, :]
    elif direction == 4:  # right-down
        matrix_shifted[:-1, :-1, :] = gene_matrix[1:, 1:, :]
        gene_weights_shifted[:-1, :-1, :] = gene_weights[1:, 1:, :]
    elif direction == 5:  # down
        matrix_shifted[:-1, :, :] = gene_matrix[1:, :, :]
        gene_weights_shifted[:-1, :, :] = gene_weights[1:, :, :]
    elif direction == 6:  # left-down
        matrix_shifted[:-1, 1:, :] = gene_matrix[1:, :-1, :]
        gene_weights_shifted[:-1, 1:, :] = gene_weights[1:, :-1, :]
    elif direction == 7:  # left
        matrix_shifted[:, 1:, :] = gene_matrix[:, :-1, :]
        gene_weights_shifted[:, 1:, :] = gene_weights[:, :-1, :]

    # Compute the gene gradient map
    gene_gradient_map = (gene_matrix - matrix_shifted) * gene_weights_shifted

    return gene_gradient_map

@njit(fastmath=True)
def compute_direction(U, V):
    angle_UV = np.arctan2(V, U)
    direction_UV = np.floor((angle_UV + 2 * np.pi) / (2 * np.pi) * 8) % 8
    direction1 = (direction_UV - 1) % 8
    direction2 = (direction_UV + 1) % 8
    return direction1, direction2

@njit(fastmath=True)
def rotate_UV(U, V, angle_rotation):
    U_rot = U * np.cos(angle_rotation) - V * np.sin(angle_rotation)
    V_rot = U * np.sin(angle_rotation) + V * np.cos(angle_rotation)
    return U_rot, V_rot

def gradient_tracking(x_coor, y_coor, U, V, bound, max_steps, decayed_weights, displacement):
    path_points_list = []

    for x, y in zip(x_coor, y_coor):
        path_point = [[x, y]]
        dx_prev, dy_prev = 0, 0
        xmin, xmax, ymin, ymax = bound

        for i in range(max_steps):
            dx = U[int(y), int(x)]
            dy = V[int(y), int(x)]

            deltas = decayed_weights[i] * np.array([-dx, dy])
            x_new, y_new = x - deltas[0], y - deltas[1]
            if not (xmin <= x_new < xmax and ymin <= y_new < ymax):
                break
            if np.sqrt(dx**2 + dy**2) * decayed_weights[i] < displacement:
                break
            path_point.append([x_new, y_new])

            x, y = x_new, y_new

        path_points_list.append(path_point)

    return path_points_list

def trace_path(paths, pixel_sums, pixel_counts):
    
    for path in paths:
        total_len = len(path)
        for idx, point in enumerate(path):
            x, y = int(point[0]), int(point[1])
            pixel_sums[y, x] += idx / len(path)
            pixel_counts[y, x] += 1

    return np.divide(pixel_sums, np.where(pixel_counts == 0, 1, pixel_counts))

def is_contained(polygons):
    if len(polygons) == 1:
        return True
    else:
        for i in range(1, len(polygons)):
            if polygons[i-1][0].contains(polygons[i][0]):
                return True

def gradient_map(
    img_cell,
    cell_box,
    index,
    gene_mtx, 
    kernel, 
    alpha=0.5, 
    expand_by=5, 
    gene=True, 
    max_steps=None, 
    ratio=1.0, 
    displacement=0.0001, 
    threshold=None,
    block_size=61,
    min_size=None):
    
    img_slice, gradient_map, gene_dense, new_box = compute_gradient_map(img_cell=img_cell, cell_box=cell_box, gene_mtx=gene_mtx, alpha=0.5, expand_by=expand_by, gene=gene)
    
    if gene_dense is not None:
            
        kernel = kernel[:, :, np.newaxis]
        gene_weights = convolve(gene_dense, kernel, mode="constant", cval=0)


        gene_gradient_js = np.zeros((gene_dense.shape[0], gene_dense.shape[1], 8))

        for i in range(8):
            # Compute the outputs for two consecutive directions
            pk = shift_and_weight(gene_dense, gene_weights, i)
            qk = shift_and_weight(gene_dense, gene_weights, (i + 1) % 8)

            # Compute Jensen-Shannon divergence
            js_divergence = scipy.spatial.distance.jensenshannon(pk, qk, axis=2)

            # Store the result
            gene_gradient_js[:, :, i] = js_divergence

        gene_gradient_js = np.nan_to_num(1 / (gene_gradient_js + 1e-8))
        gradient_map = np.concatenate([gradient_map, gene_gradient_js], axis=2)
    
    del gene_weights
    del gene_gradient_js
    del pk
    del qk
    del js_divergence
    del gene_dense
    
    
    x_direc = {0: -1, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: -1, 7: -1}
    y_direc = {0: 1, 1: 1, 2: 1, 3: 0, 4: -1, 5: -1, 6: -1, 7: 0}
    
    max_x = np.sum(np.array([(gradient_map[:, :, i]) * x_direc[i] for i in x_direc.keys()]), axis=0,)
    max_y = np.sum(np.array([(gradient_map[:, :, i]) * y_direc[i] for i in y_direc.keys()]), axis=0,)
    
    U, V = max_x / (np.max(np.abs(max_x)) + 1e-16), max_y / np.max(np.abs(max_y) + 1e-16)

    x, y = np.meshgrid(
        np.arange(gradient_map.shape[0]),
        np.arange(gradient_map.shape[1]),
        indexing="ij",
    )

    direction1, direction2 = compute_direction(U=U, V=V)

    if gradient_map.shape[2] > 7:
        gradient1 = gradient_map[x, y, direction1.astype(int)]
        gradient2 = gradient_map[x, y, direction2.astype(int)]
    else:
        gradient1 = gradient_map[x, y, direction1.astype(int)]
        gradient2 = gradient_map[x, y, direction2.astype(int)]

    angle_rotation = np.abs(gradient1 - gradient2) / (gradient1 + gradient2 + 1e-10) * (np.pi / 2) * alpha

    U, V = rotate_UV(U=U, V=V, angle_rotation=angle_rotation)
    
    max_steps = max_steps if max_steps is not None else int(max(U.shape[0], U.shape[1]) * 2 * np.pi)
    decayed_weights = np.linspace(1.5, 0.0, max_steps)
    bounds = (0, U.shape[1], 0, U.shape[0])
    
    assert (U.shape == V.shape), "The dimension of x and y component are not the same!"
    total_pixels = U.size
    n = int(total_pixels * ratio)
    pixels = np.arange(total_pixels)
    sampled_pixels = np.random.choice(pixels, size=n, replace=False)
    x_coor, y_coor = np.unravel_index(sampled_pixels, U.T.shape)
    paths = gradient_tracking(x_coor, y_coor, U, V, bounds, max_steps, decayed_weights, displacement)

    key_pixels = np.array([[img_slice[int(path[0][1]), int(path[0][0])], img_slice[int(path[-1][1]), int(path[-1][0])], path[0][0], path[0][1], path[-1][0], path[-1][-1], len(path)] for path in paths])
    try:
        num_classes = 4
        threshold_intensity = threshold_multiotsu(img_slice, num_classes)[0]
    except ValueError:
        try:
            num_classes = 3
            threshold_intensity = threshold_multiotsu(img_slice, num_classes)[0]
        except ValueError:
             threshold_intensity = 50
    try:
        num_classes = 3
        threshold_pathlen = threshold_multiotsu(key_pixels[:,6], num_classes)[0]
    except ValueError:
        num_classes = 2
        try:
            threshold_pathlen = threshold_multiotsu(key_pixels[:,6], num_classes)[0]  
        except ValueError:
            return None, index
        
    filtered_indices = (key_pixels[:, 0] >= threshold_intensity) & (key_pixels[:, 1] >= threshold_intensity) & (key_pixels[:, 6] >= threshold_pathlen)
    convergence_points = [paths[i][-1] for i in np.where(filtered_indices)[0]]
    paths = [paths[i] for i in np.where(filtered_indices)[0]]
    
    if len(paths) < 10:
        return None, index
        
    key_pixels = key_pixels[filtered_indices, :].astype(np.int32)
    
    path_msk = np.zeros_like(img_slice)
    path_msk[key_pixels[:, 3], key_pixels[:, 2]] = 255
    smoothed_img = ndi.gaussian_filter(img_slice, sigma=1)
 
    
    segmented_result = path_msk * smoothed_img
    region_image = img_slice * np.log10(segmented_result + 1)
    local_otsu = filters.threshold_local(region_image, block_size, method="gaussian")
    labeled_image, num_labels = measure.label(
            region_image > local_otsu, connectivity=2, return_num=True
        )
    
    if num_labels == 0:
            return None, index

    region_sizes = np.bincount(labeled_image.ravel())
    largest_label = region_sizes[1:].argmax() + 1
    refined_image = labeled_image == largest_label

    if min_size == None:
        min_size = np.sum(refined_image) / 2

    refined_image = convex_hull_image(ndi.binary_fill_holes(refined_image))
    
    distance = distance_transform_edt(refined_image)
    min_distance = np.min(distance[distance > 0])
    indices = np.where(distance == min_distance)
    nearest_points = list(zip(indices[0], indices[1]))
    nearest_pixel_values = np.array([img_slice[point] for point in nearest_points])
    contour_value = max(np.percentile(nearest_pixel_values, 98), np.percentile(img_slice, 68.27))
    contours = measure.find_contours(img_slice * refined_image, contour_value)
    polygons = [(make_valid(Polygon(contour)), Polygon(contour).area, contour) for contour in contours if len(contour) > 4]
    valid_polygons = [(poly, area, contour) for poly, area, contour in polygons if poly.is_valid]
    valid_polygons.sort(key=lambda x: x[1], reverse=True)
    contours_rec = [contour_value]
    
    while len(contours) == 1 or is_contained(polygons):
        if len(contours) == 1:
            contour_value += 1
            contours_rec.append(contour_value)
        elif is_contained(polygons):
            contour_value -= 1
            contours_rec.append(contour_value)
            
        contours = measure.find_contours(img_slice * refined_image, contour_value)   
        polygons = [(make_valid(Polygon(contour)), Polygon(contour).area, contour) for contour in contours if len(contour) > 4]
        valid_polygons = [(poly, area, contour) for poly, area, contour in polygons if poly.is_valid]
        valid_polygons.sort(key=lambda x: x[1], reverse=True)
        
        if len(contours_rec) >= 10:
            break

    contours = [i[2] for i in valid_polygons]

    labeled_image = np.zeros_like(img_slice, dtype=int)
    for i, contour in enumerate(contours):
        rr, cc = draw.polygon(contour[:, 0], contour[:, 1], labeled_image.shape)
        labeled_image[rr, cc] = i + 1

    regions = measure.regionprops(labeled_image)
    max_area_border = 0
    max_region_border_label = 0

    for region in regions:
        if region.area > max_area_border:
            max_area_border = region.area
            max_region_border_label = region.label

    largest_region_label = max_region_border_label

    labeled_image = (labeled_image == largest_region_label)
    labeled_image = morphology.remove_small_holes(labeled_image, area_threshold=64)
    labeled_image = morphology.remove_small_objects(labeled_image, min_size=64)

    coords_array = np.array(convergence_points, dtype=int)
    rows, cols = coords_array[:, 1], coords_array[:, 0]
    is_in_region = labeled_image[rows, cols] == 1
    convergence_points = [paths[i][-1] for i in np.where(is_in_region)[0]]
    paths = [paths[i] for i in np.where(is_in_region)[0]]
    key_pixels = key_pixels[is_in_region, :].astype(np.int32)
    path_msk = np.zeros_like(img_slice)
    path_msk[key_pixels[:, 3], key_pixels[:, 2]] = 255

    segmented_result = path_msk * smoothed_img
    region_image = img_slice * np.log10(segmented_result + 1)
    local_otsu = filters.threshold_local(region_image, block_size, method="gaussian")
    labeled_image, num_labels = measure.label(
            region_image > local_otsu, connectivity=2, return_num=True
        )

    region_sizes = np.bincount(labeled_image.ravel())

    if len(region_sizes) >= 2:
        largest_label = region_sizes[1:].argmax() + 1

        if min_size == None:
            min_size = np.sum(labeled_image == largest_label) / 2

        refined_image *= convex_hull_image(ndi.binary_fill_holes(labeled_image == largest_label))        
        
    min_x, min_y, max_x, max_y = new_box
    
    result_mask = np.zeros_like(img_cell, dtype=np.int32)
    result_mask[min_y : max_y + 1, min_x : max_x + 1][refined_image > 0] = index + 1
    if np.sum(result_mask) != 0:
        non_zero_positions = np.column_stack(np.where(result_mask > 0))
    else:
        non_zero_positions = None

    return non_zero_positions, index

def compute_bounding_box(polygon):
    """
    Get the bounding box of a polygon.
    """
    return polygon.bounds  # Returns (minx, miny, maxx, maxy)

def bounding_box_intersects(box1, box2):
    """
    Check if two bounding boxes intersect.
    """
    return not (
        box1[2] < box2[0] or  # The right boundary of box1 is to the left of the left boundary of box2
        box1[0] > box2[2] or  # The left boundary of box1 is to the right of the right boundary of box2
        box1[3] < box2[1] or  # The bottom boundary of box1 is above the top boundary of box2
        box1[1] > box2[3]     # The top boundary of box1 is below the bottom boundary of box2
    )

def check_and_merge(label, poly, polygons):
    """
    Check and merge polygons with inclusion relationships.
    """
    to_merge = [poly]
    used_labels = set()
    for other_label, other_poly in polygons.items():
        if other_label == label or other_label in used_labels:
            continue
        box1 = compute_bounding_box(poly)
        box2 = compute_bounding_box(other_poly)
        if bounding_box_intersects(box1, box2):  # Rough screening
            if other_poly.within(poly):  # Precise inclusion check
                to_merge.append(other_poly)
                used_labels.add(other_label)
            elif poly.within(other_poly):
                to_merge.append(other_poly)
                used_labels.add(label)
    return unary_union(to_merge)

def refine_mask_parallel(mask, n_jobs):
    """
    Refine the mask by checking inclusion relationships and merging.
    Args:
        mask (np.ndarray): The input mask where each object has a unique integer label.
        n_jobs (int): Number of parallel jobs for computation.

    Returns:
        np.ndarray: The refined mask.
    """
    refined_mask = np.zeros_like(mask, dtype=np.int32)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    # Store each object's pixels as polygons
    polygons = {}
    for label in unique_labels:
        coords = np.column_stack(np.where(mask == label))  # Get pixel coordinates
        polygons[label] = Polygon(coords)

    # Parallel processing for inclusion and merging
    merged_polygons = Parallel(n_jobs=n_jobs)(
        delayed(check_and_merge)(label, poly, polygons) for label, poly in polygons.items()
    )

    # Smooth and generate the new mask
    for i, merged_poly in enumerate(merged_polygons, start=1):
        if not merged_poly.is_empty:
            coords = np.array(merged_poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(refined_mask, [coords], i)

    # Morphological operations: close and smooth
    refined_mask = refined_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    return refined_mask