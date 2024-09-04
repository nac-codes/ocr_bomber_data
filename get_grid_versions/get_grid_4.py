import numpy as np
import cv2
import math
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import combinations
from tqdm import tqdm
import sys
import logging
import matplotlib.pyplot as plt
from math import atan2, degrees
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
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Vertical Line Clusters')
    plt.axis('off')
    plt.show()

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
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Horizontal Line Clusters')
    plt.axis('off')
    plt.show()

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

# Main script (example usage)
if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    vertical_lines, horizontal_lines = detect_grid_lines(image)
    
    column_widths = [
        # 292,  # First column (Day)
        129, 83, 224, 125, 240, 190, 170, 87, 81, 87, 211, 126, 257, 85, 126, 215, 121, 254, 205, 125, 205
        # , 215  # Last column (Total Tons)
    ]

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


    vertical_match, vertical_error = find_best_vertical_grid_match(vertical_lines, column_widths, image)
    horizontal_match, horizontal_error = find_best_horizontal_grid_match(horizontal_lines, row_heights, image)
    
    visualize_results(image, vertical_match, horizontal_match)
    
    print(f"Vertical match error: {vertical_error}")
    print(f"Horizontal match error: {horizontal_error}")