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


# Setup logging
logging.basicConfig(level=logging.INFO)
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

def cluster_lines_by_midpoint(lines, distance_threshold=10):
    logger.info("Clustering lines by midpoint")
    midpoints = [calculate_midpoint(line) for line in lines]
    
    if len(midpoints) < 2:
        return lines  # Return original lines if there are fewer than 2

    clusters = fcluster(linkage(midpoints), t=distance_threshold, criterion='distance')
    
    clustered_lines = [[] for _ in range(max(clusters))]
    for line, cluster in zip(lines, clusters):
        clustered_lines[cluster-1].append(line)
    
    mean_lines = []
    for cluster in clustered_lines:
        if cluster:
            # Calculate median x and y values for start and end points
            x1_median = np.median([line[0] for line in cluster])
            y1_median = np.median([line[1] for line in cluster])
            x2_median = np.median([line[2] for line in cluster])
            y2_median = np.median([line[3] for line in cluster])
            
            mean_line = [x1_median, y1_median, x2_median, y2_median]
            mean_lines.append(mean_line)
    
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
    
    # Display the clusters of vertical lines
    debug_image = image.copy()
    for line in clustered_lines:
        x1, y1, x2, y2 = line
        cv2.line(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Show the image with the clusters
    # plt.figure(figsize=(20, 10))
    # plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Vertical Line Clusters')
    # plt.axis('off')
    # plt.show()

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
    
    # Display the clusters of horizontal lines
    debug_image = image.copy()
    for line in clustered_lines:
        x1, y1, x2, y2 = line
        cv2.line(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
    
    # Show the image with the clusters
    # plt.figure(figsize=(20, 10))
    # plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Horizontal Line Clusters')
    # plt.axis('off')
    # plt.show()

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
            cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3]),), color, 2)

    # Draw matched horizontal grid lines
    if horizontal_match:
        for i, line in enumerate(horizontal_match):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(horizontal_match)-1 else (255, 255, 0)
            # draw the whole line
            cv2.line(result, (int(line[0]), int(line[1])), (int(line[2]), int(line[3]),), color, 2)

    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Grid Match with Orientation')
    plt.axis('off')
    plt.show()

# def splice_image(image, vertical_lines, horizontal_lines, column_widths, row_heights, output_dir):
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Sort lines by x and y coordinates
#     vertical_lines = sorted(vertical_lines, key=lambda line: (line[0] + line[2]) / 2)
#     horizontal_lines = sorted(horizontal_lines, key=lambda line: (line[1] + line[3]) / 2)

#     # Interpolate missing first and last vertical lines
#     image_width = image.shape[1]
#     image_height = image.shape[0]

#     # Calculate scaling factor
#     detected_width = vertical_lines[-1][0] - vertical_lines[0][0]
#     expected_width = sum(column_widths[1:-1])  # Exclude first and last columns
#     scale_factor = detected_width / expected_width

#     # Interpolate first vertical line (Day column)
#     first_line_x = vertical_lines[0][0] - column_widths[0] * scale_factor
#     first_line = [first_line_x, 0, first_line_x, image_height]
#     vertical_lines.insert(0, first_line)

#     # Interpolate last vertical line (Total Tons column)
#     last_line_x = vertical_lines[-1][2] + column_widths[-1] * scale_factor
#     last_line = [last_line_x, 0, last_line_x, image_height]
#     vertical_lines.append(last_line)

#     # Define column names
#     column_names = [
#         "Day", "Month", "Year", "Time_of_Attack", "Air_Force", "Group_Squadron_Number",
#         "Number_of_Aircraft_Bombing", "Altitude_of_Release", "Sighting", "Visibility_of_Target",
#         "Target_Priority", "HE_Bombs_Number", "HE_Bombs_Size", "HE_Bombs_Tons",
#         "Fuzing_Nose", "Fuzing_Tail", "Incendiary_Bombs_Number", "Incendiary_Bombs_Size",
#         "Incendiary_Bombs_Tons", "Fragmentation_Bombs_Number", "Fragmentation_Bombs_Size",
#         "Fragmentation_Bombs_Tons", "Total_Tons"
#     ]

#     # Specific regions
#     regions = [
#         ("Target_Location", 2, 0, 2, 7),  # Row 3, Columns 1-8
#         ("Target_Name", 3, 0, 3, 7),  # Row 4, Columns 1-8
#         ("Latitude", 2, 10, 2, 13),  # Row 3, Columns 11-14
#         ("Longitude", 2, 13, 2, 15),  # Row 3, Columns 14-16
#         ("Target_Code_Part1", 2, 16, 2, 16),  # Row 3, Column 17
#         ("Target_Code_Part2", 2, 17, 2, 19),  # Row 3, Columns 18-20
#     ]

#     # Function to get coordinates for a cell
#     def get_cell_coords(row, col):
#         x1, y1, _, _ = vertical_lines[col]
#         _, y2, x2, _ = vertical_lines[col + 1]
#         top = horizontal_lines[row][1]
#         bottom = horizontal_lines[row + 1][3]
#         return int(x1), int(top), int(x2), int(bottom)

#     # Splice specific regions
#     for name, start_row, start_col, end_row, end_col in regions:
#         logger.info(f"Splicing region: {name}")
#         x1, y1, _, _ = get_cell_coords(start_row, start_col)
#         logger.info(f"Coordinates: {x1}, {y1}")
#         _, _, x2, y2 = get_cell_coords(end_row, end_col)
#         logger.info(f"Coordinates: {x2}, {y2}")
#         region = image[y1:y2, x1:x2]
#         # display the region
#         plt.figure(figsize=(10, 5))
#         plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
#         plt.title(f"Region: {name}")
#         plt.axis('off')
#         plt.show()
        
#         cv2.imwrite(os.path.join(output_dir, f"{name}.png"), region)

#     # Splice data rows
#     for row in range(7, len(horizontal_lines) - 1):  # Start from row 8 (index 7)
#         for col, name in enumerate(column_names):
#             x1, y1, x2, y2 = get_cell_coords(row, col)
#             cell = image[y1:y2, x1:x2]
#             cv2.imwrite(os.path.join(output_dir, f"entry_{row-6}_{name}.png"), cell)

#     print(f"Images saved in {output_dir}")

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

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None  # Lines are parallel
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / det
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / det
    return int(px), int(py)

def four_point_transform(image, pts):
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

    # Construct set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

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

    regions = [
        ("Target_Location", 2, 0, 2, 7),
        ("Target_Name", 3, 0, 3, 7),
        ("Latitude", 2, 10, 2, 13),
        ("Longitude", 2, 13, 2, 15),
        ("Target_Code_Part1", 2, 16, 2, 16),
        ("Target_Code_Part2", 2, 17, 2, 19),
    ]

    def get_cell_polygon(row, col):
        top_left = line_intersection(vertical_lines[col], horizontal_lines[row])
        top_right = line_intersection(vertical_lines[col+1], horizontal_lines[row])
        bottom_right = line_intersection(vertical_lines[col+1], horizontal_lines[row+1])
        bottom_left = line_intersection(vertical_lines[col], horizontal_lines[row+1])
        return [top_left, top_right, bottom_right, bottom_left]

    def save_region(name, polygon):
        if any(pt is None for pt in polygon):
            logger.warning(f"Invalid polygon for {name}: {polygon}")
            return
        region = four_point_transform(image, polygon)
        if region.size == 0:
            logger.warning(f"Empty region for {name}")
            return
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), region)
        logger.info(f"Saved {name}.png")

    for name, start_row, start_col, end_row, end_col in regions:
        logger.info(f"Splicing region: {name}")
        polygon = [
            line_intersection(vertical_lines[start_col], horizontal_lines[start_row]),
            line_intersection(vertical_lines[end_col+1], horizontal_lines[start_row]),
            line_intersection(vertical_lines[end_col+1], horizontal_lines[end_row+1]),
            line_intersection(vertical_lines[start_col], horizontal_lines[end_row+1])
        ]
        save_region(name, polygon)

    for row in range(7, len(horizontal_lines) - 1):
        for col, name in enumerate(column_names):
            polygon = get_cell_polygon(row, col)
            save_region(f"entry_{row-6}_{name}", polygon)

    logger.info(f"Image splicing completed. Results saved in {output_dir}")

# Main script (example usage)
if __name__ == "__main__":
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    vertical_lines, horizontal_lines = detect_grid_lines(image)
    
    column_widths = [
        # 292,  # First column (Day)
        129, 83, 224, 125, 240, 190, 170, 87, 81, 87, 211, 126, 257, 85, 126, 215, 121, 254, 205, 125, 205
        # , 215  # Last column (Total Tons)
    ]
    # print out the number of columns
    print(f"Number of columns: {len(column_widths)}")

    # row_heights = [
    # 93, 93, 92, 89, 30,  # Rows, Blank, Target, Location, Buffer
    # 58, 32, 55, 33, 56,  # Header Rows
    # 90, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87  # Data Rows
    # ]

    row_heights = [
    93, 93, 92, 89,  # Rows, Blank, Target, Location, Buffer
    58, 55, 56,  # Header Rows
    90, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87  # Data Rows
    ]
    print(f"Number of rows: {len(row_heights)}")


    vertical_match, vertical_error = find_best_vertical_grid_match(vertical_lines, column_widths, image)
    horizontal_match, horizontal_error = find_best_horizontal_grid_match(horizontal_lines, row_heights, image)
    
    visualize_results(image, vertical_match, horizontal_match)
    
    print(f"Vertical match error: {vertical_error}")
    print(f"Horizontal match error: {horizontal_error}")

    # check if the correct number of vertical and horizontal lines were found, if not exit with error code 1
    if len(vertical_match) != len(column_widths) + 1 or len(horizontal_match) != len(row_heights) + 1:
        # print the number of vertical and horizontal lines found
        logger.error(f"Incorrect number of vertical or horizontal lines found: {len(vertical_match)}, {len(horizontal_match)}")
        sys.exit(1)

    # Splice the image
    output_dir = image_path.replace('.JPG', '_output')
    splice_image(image, vertical_match, horizontal_match, column_widths, row_heights, output_dir)