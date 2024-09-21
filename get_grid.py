import numpy as np
import cv2
import math
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import combinations
from tqdm import tqdm
import sys
import os
import logging
import matplotlib.pyplot as plt
from math import atan2, degrees
from scipy.stats import linregress
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label


# Setup logging
logging.basicConfig(filename='grid_getting_log.txt', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_grid_lines(image, visualize=True):
    logger.info("Starting target image line detection")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

    if lines is None:
        logger.warning("No lines detected in the target image")
        return []

    lines = lines.reshape(-1, 4)

    def line_angle(line):
        return degrees(atan2(line[3] - line[1], line[2] - line[0])) % 180

    angles = np.array([line_angle(line) for line in lines])

    # Cluster lines based on their angle
    angle_clusters = fcluster(linkage(angles.reshape(-1, 1)), t=5, criterion='distance')

    vertical_lines = []
    horizontal_lines = []

    for cluster in range(1, angle_clusters.max() + 1):
        cluster_lines = lines[angle_clusters == cluster]
        mean_angle = np.mean([line_angle(line) for line in cluster_lines])

        if 45 <= mean_angle <= 135:
            vertical_lines.extend(cluster_lines)
        else:
            horizontal_lines.extend(cluster_lines)


    logger.info(f"Detected {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines in the target image")
    return vertical_lines, horizontal_lines

# ------ Extending Functions ------

def extend_line_vertical(line, image_height):
    x1, y1, x2, y2 = line
    if y2 != y1:
        slope = (x2 - x1) / (y2 - y1)
        x1_full = x1 - slope * y1
        x2_full = x1 + slope * (image_height - y1)
        return [x1_full, 0, x2_full, image_height]
    else:
        return [x1, 0, x1, image_height]

def extend_line_horizontal(line, image_width):
    x1, y1, x2, y2 = line
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        y1_full = y1 - slope * x1
        y2_full = y1 + slope * (image_width - x1)
        return [0, y1_full, image_width, y2_full]
    else:
        return [0, y1, image_width, y1]

# ------ Parallel Line Functions ------

def calculate_line_angle(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

def find_parallel_lines(lines, angle_threshold=1):
    logger.info("Finding parallel lines")
    angles = [calculate_line_angle(line) for line in lines]
    angle_clusters = fcluster(linkage(np.array(angles).reshape(-1, 1)), t=angle_threshold, criterion='distance')
    
    largest_cluster = max(set(angle_clusters), key=list(angle_clusters).count)
    parallel_lines = [line for line, cluster in zip(lines, angle_clusters) if cluster == largest_cluster]

    logger.info(f"Found {len(parallel_lines)} parallel lines")
    
    return parallel_lines

def calculate_midpoint(line):
    x1, y1, x2, y2 = line
    return (x1 + x2) / 2, (y1 + y2) / 2

def adjust_line_to_angle(line, mean_angle_degrees):
    x_mid, y_mid = calculate_midpoint(line)
    length = math.hypot(line[2] - line[0], line[3] - line[1])
    mean_angle_radians = math.radians(mean_angle_degrees)
    dx = (length / 2) * math.cos(mean_angle_radians)
    dy = (length / 2) * math.sin(mean_angle_radians)
    x1_new = x_mid - dx
    y1_new = y_mid - dy
    x2_new = x_mid + dx
    y2_new = y_mid + dy
    return [x1_new, y1_new, x2_new, y2_new]

def cluster_lines_by_midpoint(lines, distance_threshold=10):
    logger.info("Clustering lines by midpoint and adjusting to mean angle")
    midpoints = [calculate_midpoint(line) for line in lines]
    
    if len(midpoints) < 2:
        return lines  # Return original lines if there are fewer than 2

    clusters = fcluster(linkage(midpoints), t=distance_threshold, criterion='distance')
    
    clustered_lines = [[] for _ in range(max(clusters))]
    for line, cluster in zip(lines, clusters):
        clustered_lines[cluster-1].append(line)
    
    mean_lines = []
    for idx, cluster in enumerate(clustered_lines):
        if cluster:
            # Calculate mean angle
            angles = [calculate_line_angle(line) for line in cluster]
            mean_angle = np.mean(angles)

            # Adjust lines to have mean angle, preserving midpoint
            adjusted_lines = [adjust_line_to_angle(line, mean_angle) for line in cluster]

            # Compute median x and y values for start and end points of adjusted lines
            x1_median = np.median([line[0] for line in adjusted_lines])
            y1_median = np.median([line[1] for line in adjusted_lines])
            x2_median = np.median([line[2] for line in adjusted_lines])
            y2_median = np.median([line[3] for line in adjusted_lines])

            mean_line = [x1_median, y1_median, x2_median, y2_median]
            mean_lines.append(mean_line)
            logger.debug(f"Cluster {idx+1}: Mean angle adjusted to {mean_angle:.2f} degrees")

    logger.info(f"Found {len(mean_lines)} clusters")
    return mean_lines

# ------ Grid Matching Functions ------

def find_best_vertical_grid_match(vertical_lines, column_widths, image):
    logger.info("Finding best vertical grid match")

    image_height = image.shape[0]

    # Extend all lines to image_height
    logger.info("Extending vertical lines to image height")
    vertical_lines = [extend_line_vertical(line, image_height) for line in vertical_lines]

    parallel_lines = find_parallel_lines(vertical_lines)
    clustered_lines = cluster_lines_by_midpoint(parallel_lines)
    
    # check if the number of vertical lines is less than the number of columns
    if len(clustered_lines) < len(column_widths) + 1:
        logger.warning("Number of vertical lines is less than the number of columns")
        return [], float('inf')

    # Calculate the expected ratios
    total_width = sum(column_widths)
    expected_ratios = np.array(column_widths) / total_width

    # Sort clustered lines by x-coordinate
    sorted_lines = sorted(clustered_lines, key=lambda line: (line[0] + line[2]) / 2)

    def calculate_error(start_idx, end_idx, ratio):
        if start_idx >= end_idx:
            return float('inf')
        start_x = (sorted_lines[start_idx][0] + sorted_lines[start_idx][2]) / 2
        end_x = (sorted_lines[end_idx][0] + sorted_lines[end_idx][2]) / 2
        actual_ratio = (end_x - start_x) / total_width
        return (actual_ratio - ratio) ** 2

    n = len(sorted_lines)
    m = len(column_widths)
    dp = [[float('inf')] * (m + 1) for _ in range(n)]
    prev = [[None] * (m + 1) for _ in range(n)]

    # Initialize first column
    for i in range(n):
        dp[i][0] = 0

    # Dynamic programming
    for j in range(1, m + 1):
        for i in range(j, n):
            for k in range(j - 1, i):
                error = dp[k][j-1] + calculate_error(k, i, expected_ratios[j-1])
                if error < dp[i][j]:
                    dp[i][j] = error
                    prev[i][j] = k

    # Find the best end point
    best_end = min(range(m, n), key=lambda i: dp[i][m])
    best_error = dp[best_end][m]

    # Reconstruct the best match
    best_match = []
    curr = best_end
    for j in range(m, 0, -1):
        best_match.append(sorted_lines[curr])
        curr = prev[curr][j]
    best_match.append(sorted_lines[curr])
    best_match.reverse()

    logger.info(f"Best vertical match found with error: {best_error}")
    return best_match, best_error

def find_best_horizontal_grid_match(horizontal_lines, row_heights, image):
    logger.info("Finding best horizontal grid match")

    image_width = image.shape[1]

    # Extend all lines to image_width
    logger.info("Extending horizontal lines to image width")
    horizontal_lines = [extend_line_horizontal(line, image_width) for line in horizontal_lines]

    parallel_lines = find_parallel_lines(horizontal_lines)
    clustered_lines = cluster_lines_by_midpoint(parallel_lines)
    
    # Check if the number of horizontal lines is less than the number of rows
    if len(clustered_lines) < len(row_heights):
        logger.warning("Number of horizontal lines is less than the number of rows")
        return [], float('inf')

    # Calculate the expected ratios
    total_height = sum(row_heights)
    expected_ratios = np.array(row_heights) / total_height

    # Sort clustered lines by y-coordinate
    sorted_lines = sorted(clustered_lines, key=lambda line: (line[1] + line[3]) / 2)

    def calculate_error(start_idx, end_idx, ratio):
        if start_idx >= end_idx:
            return float('inf')
        start_y = (sorted_lines[start_idx][1] + sorted_lines[start_idx][3]) / 2
        end_y = (sorted_lines[end_idx][1] + sorted_lines[end_idx][3]) / 2
        actual_ratio = (end_y - start_y) / total_height
        return (actual_ratio - ratio) ** 2

    n = len(sorted_lines)
    m = len(row_heights)
    dp = [[float('inf')] * (m + 1) for _ in range(n)]
    prev = [[None] * (m + 1) for _ in range(n)]

    # Initialize first column
    for i in range(n):
        dp[i][0] = 0

    # Dynamic programming
    for j in tqdm(range(1, m + 1), desc="Processing rows"):
        for i in range(j, n):
            for k in range(j - 1, i):
                error = dp[k][j-1] + calculate_error(k, i, expected_ratios[j-1])
                if error < dp[i][j]:
                    dp[i][j] = error
                    prev[i][j] = k

    # Find the best end point
    best_end = min(range(m, n), key=lambda i: dp[i][m])
    best_error = dp[best_end][m]

    # Reconstruct the best match
    best_match = []
    curr = best_end
    for j in range(m, 0, -1):
        best_match.append(sorted_lines[curr])
        curr = prev[curr][j]
    best_match.append(sorted_lines[curr])
    best_match.reverse()

    logger.info(f"Best horizontal match found with error: {best_error}")
    return best_match, best_error

# ------ Visualization Function ------

def visualize_results(image, vertical_match, horizontal_match):
    result = image.copy()
    
    # Draw matched vertical grid lines
    if vertical_match:
        for i, line in enumerate(vertical_match):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(vertical_match)-1 else (255, 0, 0)
            # draw the whole line
            cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color, 2)

    # Draw matched horizontal grid lines
    if horizontal_match:
        for i, line in enumerate(horizontal_match):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(horizontal_match)-1 else (255, 255, 0)
            # draw the whole line
            cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color, 2)

    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Grid Match with Orientation')
    plt.axis('off')
    plt.show()


def add_boundary_columns(vertical_lines, column_widths, image_shape):
    """
    Add the first and last columns to the vertical lines based on the provided column widths.
    
    Args:
    vertical_lines (list): List of vertical lines, each represented as [x1, y1, x2, y2].
    column_widths (list): List of column widths including the first and last columns.
    image_shape (tuple): Shape of the image (height, width).
    
    Returns:
    list: Updated list of vertical lines including the interpolated first and last columns.
    """
    image_height, image_width = image_shape[:2]
    
    # Sort vertical lines
    vertical_lines = sorted(vertical_lines, key=lambda line: (line[0] + line[2]) / 2)
    
    # Calculate scaling factor
    detected_width = vertical_lines[-1][0] - vertical_lines[0][0]
    expected_width = sum(column_widths[1:-1])  # Exclude first and last columns
    scale_factor = detected_width / expected_width
    
    logger.info(f"Detected width: {detected_width}, Expected width: {expected_width}")
    logger.info(f"Scale factor: {scale_factor}")
    
    # Interpolate first vertical line (Day column)
    first_line_x = max(0, vertical_lines[0][0] - column_widths[0] * scale_factor)
    first_line = [first_line_x, 0, first_line_x, image_height]
    
    # Interpolate last vertical line (Total Tons column)
    last_line_x = min(image_width, vertical_lines[-1][2] + column_widths[-1] * scale_factor)
    last_line = [last_line_x, 0, last_line_x, image_height]
    
    # Add new lines to the list
    updated_vertical_lines = [first_line] + vertical_lines + [last_line]
    
    logger.info(f"Added boundary columns. Total vertical lines: {len(updated_vertical_lines)}")
    
    return updated_vertical_lines

def adjust_lines_to_median_slope(match, deviation_threshold=1):
    """
    Adjusts lines to conform to the median slope if they deviate too much.
    
    Args:
    match (list): List of lines, each represented as [x1, y1, x2, y2].
    deviation_threshold (float): Number of standard deviations from median to allow.
    
    Returns:
    list: Adjusted list of lines with slopes conforming to the median slope.
    """
    # Calculate the slope and midpoint of each line
    slopes = []
    midpoints = []
    for line in match:
        x1, y1, x2, y2 = line
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = float('inf')  # Vertical line
        slopes.append(slope)
        midpoints.append(((x1 + x2) / 2, (y1 + y2) / 2))
    
    # Calculate median slope and standard deviation
    median_slope = np.median(slopes)
    slope_std = np.std(slopes)
    
    # Adjust the lines that deviate too much from the median slope
    adjusted_match = []
    for (x_mid, y_mid), slope in zip(midpoints, slopes):
        if abs(slope - median_slope) > deviation_threshold * slope_std:
            # Use the median slope to calculate new endpoints
            if median_slope != float('inf'):
                half_length = np.sqrt(((match[0][2] - match[0][0])**2 + (match[0][3] - match[0][1])**2) / 4)
                dx = half_length / np.sqrt(1 + median_slope**2)
                dy = median_slope * dx
                new_line = [x_mid - dx, y_mid - dy, x_mid + dx, y_mid + dy]
            else:
                # For vertical lines
                half_height = (match[0][3] - match[0][1]) / 2
                new_line = [x_mid, y_mid - half_height, x_mid, y_mid + half_height]
        else:
            # Keep the original line if it's within the deviation threshold
            new_line = [x_mid - (match[0][2] - match[0][0])/2, 
                        y_mid - (match[0][3] - match[0][1])/2,
                        x_mid + (match[0][2] - match[0][0])/2, 
                        y_mid + (match[0][3] - match[0][1])/2]
        
        adjusted_match.append(new_line)
    
    return adjusted_match

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None  # Lines are parallel
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / det
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / det
    return int(px), int(py)

def four_point_transform(image, pts, padding_percentage=0.15):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Calculate padding
    padding = int(maxHeight * padding_percentage)
    padded_height = maxHeight + 2 * padding

    # Construct set of destination points with padding
    dst = np.array([
        [0, padding],
        [maxWidth - 1, padding],
        [maxWidth - 1, padded_height - padding - 1],
        [0, padded_height - padding - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, padded_height))

    return warped

def remove_vertical_lines(binary_image):
    """
    Remove vertical lines from a binary image to avoid interference with histogram calculations.
    """
    # Ensure the image is large enough for processing
    min_height = 100  # Minimum height to apply vertical line removal
    if binary_image.shape[0] < min_height:
        return binary_image  # Return original image if too small

    # Create a kernel for vertical lines
    kernel_height = max(1, binary_image.shape[0] // 100)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    
    # Detect vertical lines
    detected_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    # Subtract detected lines from the binary image
    processed_image = cv2.subtract(binary_image, detected_lines)
    return processed_image

def adjust_cell_boundaries(cell_image):
    """
    Adjust the top and bottom boundaries of the cell image to fully contain the number without cutting it off.
    Uses a more lenient dynamic threshold based on image statistics.
    """
    logger.debug("Starting boundary adjustment for cell")
    # Convert to grayscale
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate image statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Calculate dynamic threshold for binary image
    # Adjusted to be more lenient (closer to the mean)
    dynamic_threshold = mean_intensity - 0.8 * std_intensity
    dynamic_threshold = max(0, min(255, dynamic_threshold))  # Ensure it's within [0, 255]
    
    logger.debug(f"Dynamic threshold for binary image: {dynamic_threshold}")
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, dynamic_threshold, 255, cv2.THRESH_BINARY_INV)
    
    try:
        # Optionally remove vertical lines
        binary = remove_vertical_lines(binary)
    except Exception as e:
        logger.warning(f"Error in removing vertical lines: {e}. Proceeding with original binary image.")
    
    # Compute the proportion of black pixels in each row
    black_proportion = np.sum(binary == 255, axis=1) / binary.shape[1]
    
    # Calculate dynamic black threshold
    mean_black_prop = np.mean(black_proportion)
    std_black_prop = np.std(black_proportion)
    
    # Adjusted to be more lenient
    black_threshold = mean_black_prop + 0.3 * std_black_prop
    black_threshold = max(0.01, min(0.1, black_threshold))  # Limit range to [0.01, 0.1]
    
    logger.debug(f"Dynamic black threshold: {black_threshold}")
    
    # Find rows where the proportion of black pixels is above the threshold
    black_rows = black_proportion > black_threshold

    # Log the black proportion per row for debugging
    for idx, prop in enumerate(black_proportion):
        bars = '|' * int(prop * 20)  # Visual representation
        spaces = '-' * (20 - int(prop * 20))
        row_type = 'Black' if black_rows[idx] else 'White'
        logger.debug(f"Row {idx}: {row_type} {bars}{spaces}")

    # Find contiguous regions of black rows
    labeled_array, num_features = label(black_rows)
    # If no black regions are found, return None
    if num_features == 0:
        logger.debug("No black regions found in cell.")
        return None

    # Find the largest contiguous region (the main content)
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # Background is label 0
    max_label = np.argmax(sizes)
    max_size = sizes[max_label]
    main_region = labeled_array == max_label
    main_indices = np.where(main_region)[0]
    top_row = main_indices[0]
    bottom_row = main_indices[-1]

    logger.debug(f"Main black region size: {max_size} rows (from row {top_row} to row {bottom_row})")

    # threshold of rows required based on number of rows
    min_rows = max(int(.15 * binary.shape[0]), 7)  # At least 7 rows or 15% of image height

    # If the largest cluster is smaller than the minimum, return None
    if max_size < min_rows:
        logger.debug(f"Largest black region is less than {min_rows} rows. Ignoring cell.")
        return None

    # Now, check for at least two white rows above and below
    min_white_rows = 2

    # Check if the top row has two white rows above it
    if top_row - min_white_rows < 0:
        # Check if the main region takes up at least 30% of the image height
        if (bottom_row - top_row) / binary.shape[0] > 0.3:
            adjusted_top = 0
        else:
            logger.debug("Not enough white rows above the main region. Ignoring cell.")
            return None
    else:
        adjusted_top = top_row - min_white_rows

    # Check if the bottom row has two white rows below it
    if bottom_row + min_white_rows >= binary.shape[0]:
        if (bottom_row - top_row) / binary.shape[0] > 0.3:
            adjusted_bottom = binary.shape[0]
        else:
            logger.debug("Not enough white rows below the main region. Ignoring cell.")
            return None
    else:
        adjusted_bottom = bottom_row + min_white_rows

    logger.debug(f"Adjusted content region: adjusted_top={adjusted_top}, adjusted_bottom={adjusted_bottom}")

    # Crop the image
    adjusted_image = cell_image[adjusted_top:adjusted_bottom, :]

    # padded_image = add_white_padding(adjusted_image, padding_top=15, padding_bottom=15, padding_left=10, padding_right=10)
    padded_image = adjusted_image


    return padded_image

def splice_image(image, vertical_lines, horizontal_lines, column_widths, row_heights, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_height, image_width = image.shape[:2]
    logger.info(f"Image dimensions: {image_width}x{image_height}")

    # Add boundary columns
    vertical_lines = add_boundary_columns(vertical_lines, column_widths, image.shape)

    # Sort horizontal lines
    horizontal_lines = sorted(horizontal_lines, key=lambda line: (line[1] + line[3]) / 2)

    column_names = [
        "Day", "Month", "Year", "Time_of_Attack", "Air_Force", "Group_Squadron_Number",
        "Number_of_Aircraft_Bombing", "Altitude_of_Release", "Sighting", "Visibility_of_Target",
        "Target_Priority", "HE_Bombs_Number", "HE_Bombs_Size", "HE_Bombs_Tons",
        "Fuzing_Nose", "Fuzing_Tail", "Incendiary_Bombs_Number", "Incendiary_Bombs_Size",
        "Incendiary_Bombs_Tons", "Fragmentation_Bombs_Number", "Fragmentation_Bombs_Size",
        "Fragmentation_Bombs_Tons", "Total_Tons"
    ]

    # Adjusted regions as per your instructions
    regions = [
        ("Target_Location", 0, 0, 2, 7),
        ("Target_Name", 3, 0, 5, 9),
        ("Latitude", 0, 10, 5, 12),
        ("Longitude", 0, 13, 5, 15),
        ("Target_Code", 1, 17, 5, 21)
    ]

    def get_cell_polygon(row, col):
        top_left = line_intersection(vertical_lines[col], horizontal_lines[row])
        top_right = line_intersection(vertical_lines[col+1], horizontal_lines[row])
        bottom_right = line_intersection(vertical_lines[col+1], horizontal_lines[row+1])
        bottom_left = line_intersection(vertical_lines[col], horizontal_lines[row+1])
        return [top_left, top_right, bottom_right, bottom_left]

    def split_polygon(polygon):
        # Calculate midpoints
        left_midpoint = ((polygon[0][0] + polygon[3][0]) / 2, (polygon[0][1] + polygon[3][1]) / 2)
        right_midpoint = ((polygon[1][0] + polygon[2][0]) / 2, (polygon[1][1] + polygon[2][1]) / 2)
        
        # Create two new polygons
        top_polygon = [polygon[0], polygon[1], right_midpoint, left_midpoint]
        bottom_polygon = [left_midpoint, right_midpoint, polygon[2], polygon[3]]
        
        return top_polygon, bottom_polygon

    def save_region(name, polygon, counter, split=True, adjust_boundaries=True):
        if any(pt is None for pt in polygon):
            logger.warning(f"Invalid polygon for {name}: {polygon}")
            return counter

        if split:
            top_polygon, bottom_polygon = split_polygon(polygon)
            polygons = [top_polygon, bottom_polygon]
        else:
            polygons = [polygon]

        for i, poly in enumerate(polygons):
            region = four_point_transform(image, poly, padding_percentage=0.25)
            if region.size == 0:
                logger.warning(f"Empty region for {name} (part {i+1})")
                continue

            if adjust_boundaries:
                # save a version of the region without adjusted boundaries for debugging
                # cv2.imwrite(os.path.join(output_dir, f"{counter:03d}_{name}_part{i+1}_original.png"), region)

                # Adjust cell boundaries
                adjusted_region = adjust_cell_boundaries(region)
                if adjusted_region is None:
                    logger.info(f"No significant content found in {name} (part {i+1}). Skipping save.")
                    continue
            else:
                adjusted_region = region

            if split:
                filename = f"{counter:03d}_{name}_part{i+1}.png"
            else:
                filename = f"{counter:03d}_{name}.png"

            cv2.imwrite(os.path.join(output_dir, filename), adjusted_region)
            logger.info(f"Saved {filename}")
            counter += 1

        return counter

    counter = 1

    # Process header regions without splitting or boundary adjustment
    for name, start_row, start_col, end_row, end_col in regions:
        logger.info(f"Splicing region: {name}")
        polygon = [
            line_intersection(vertical_lines[start_col], horizontal_lines[start_row]),
            line_intersection(vertical_lines[end_col+1], horizontal_lines[start_row]),
            line_intersection(vertical_lines[end_col+1], horizontal_lines[end_row+1]),
            line_intersection(vertical_lines[start_col], horizontal_lines[end_row+1])
        ]
        counter = save_region(name, polygon, counter, split=False, adjust_boundaries=False)

    # Process data cells with splitting and boundary adjustment
    for row in range(7, len(horizontal_lines) - 1):
        for col, name in enumerate(column_names):
            polygon = get_cell_polygon(row, col)
            counter = save_region(f"entry_{row-6}_{name}", polygon, counter)

    logger.info(f"Image splicing completed. Results saved in {output_dir}")

# Main script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("No image path provided.")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image at path: {image_path}")
        sys.exit(1)
    logger.info(f"Processing image: {image_path}")
    
    vertical_lines, horizontal_lines = detect_grid_lines(image)
    
    column_widths = [
        # 292,  # First column (Day)
        129, 83, 224, 125, 240, 190, 170, 87, 81, 87, 211, 126, 257, 85, 126, 215, 121, 254, 205, 125, 205
        # , 215  # Last column (Total Tons)
    ]
    # print out the number of columns
    logger.info(f"Number of columns: {len(column_widths)}")

    # Adjusted row heights as per your requirements
    row_heights = [
        93, 93, 92, 89,  # Rows: Blank, Target, Location, Buffer
        58, 55, 56,      # Header Rows
        90, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87  # Data Rows
    ]
    logger.info(f"Number of rows: {len(row_heights)}")

    vertical_match, vertical_error = find_best_vertical_grid_match(vertical_lines, column_widths, image)
    horizontal_match, horizontal_error = find_best_horizontal_grid_match(horizontal_lines, row_heights, image)

    # Adjust the vertical match to have the same slope
    vertical_match = adjust_lines_to_median_slope(vertical_match)
    horizontal_match = adjust_lines_to_median_slope(horizontal_match)
    
    visualize_results(image, vertical_match, horizontal_match)
    
    logger.info(f"Vertical match error: {vertical_error}")
    logger.info(f"Horizontal match error: {horizontal_error}")

    # check if the correct number of vertical and horizontal lines were found, if not exit with error code 1
    if len(vertical_match) != len(column_widths) + 1 or len(horizontal_match) != len(row_heights) + 1:
        # print the number of vertical and horizontal lines found
        logger.error(f"Incorrect number of vertical or horizontal lines found: {len(vertical_match)}, {len(horizontal_match)}")
        sys.exit(1)

    # Splice the image
    output_dir = os.path.splitext(image_path)[0] + '_output'
    splice_image(image, vertical_match, horizontal_match, column_widths, row_heights, output_dir)
