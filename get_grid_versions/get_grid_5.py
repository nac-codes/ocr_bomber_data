import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from math import atan2, degrees
from scipy.cluster.hierarchy import linkage, fcluster
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
        return [], []

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

def find_best_grid_match(lines, widths, is_vertical):
    logger.info(f"Finding best {'vertical' if is_vertical else 'horizontal'} grid match")
    target_count = len(widths) + 1
    total_width = sum(widths)
    normalized_widths = np.array([width / total_width for width in widths])
    
    # Sort lines by x-coordinate (for vertical) or y-coordinate (for horizontal)
    sorted_lines = sorted(lines, key=lambda l: l[0] if is_vertical else l[1])
    
    # Calculate all possible widths
    all_widths = []
    for i in range(len(sorted_lines) - 1):
        for j in range(i + 1, min(i + target_count, len(sorted_lines))):
            width = sorted_lines[j][0 if is_vertical else 1] - sorted_lines[i][0 if is_vertical else 1]
            all_widths.append((i, j, width))
    
    # Normalize widths
    total_calculated_width = sum(w for _, _, w in all_widths)
    normalized_calculated_widths = np.array([w / total_calculated_width for _, _, w in all_widths])
    
    # Create cost matrix
    cost_matrix = np.abs(normalized_calculated_widths[:, np.newaxis] - normalized_widths)
    
    # Use the Hungarian algorithm to find the best assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Reconstruct the best matching lines
    best_match = []
    for i, j, _ in [all_widths[i] for i in row_ind]:
        if not best_match or i != best_match[-1]:  # Avoid duplicates
            best_match.append(i)
        best_match.append(j)
    
    # Remove duplicates while preserving order
    best_match = list(dict.fromkeys(best_match))
    
    # Ensure we have exactly target_count lines
    if len(best_match) > target_count:
        best_match = best_match[:target_count]
    elif len(best_match) < target_count:
        # Add more lines if we don't have enough
        for i in range(len(sorted_lines)):
            if i not in best_match:
                best_match.append(i)
                if len(best_match) == target_count:
                    break
    
    # Convert indices back to actual lines
    best_match = [sorted_lines[i] for i in best_match]
    
    # Calculate the error for the best match
    best_error = calculate_error(best_match, widths, is_vertical)
    
    logger.info(f"Best match found with error: {best_error}")
    return best_match, best_error

def calculate_error(lines, widths, is_vertical):
    if len(lines) != len(widths) + 1:
        return float('inf')
    
    total_width = sum(widths)
    normalized_widths = np.array([width / total_width for width in widths])
    
    actual_widths = np.diff([line[0 if is_vertical else 1] for line in lines])
    total_actual_width = np.sum(actual_widths)
    normalized_actual_widths = actual_widths / total_actual_width
    
    return np.sum((normalized_widths - normalized_actual_widths) ** 2)

def visualize_results(image, vertical_lines, horizontal_lines, vertical_match, horizontal_match):
    result = image.copy()
    
    # Draw matched vertical grid lines
    if vertical_match:
        for i, line in enumerate(vertical_match):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(vertical_match)-1 else (255, 0, 0)
            cv2.line(result, (int(line[0]), 0), (int(line[0]), image.shape[0]), color, 2)

    # Draw matched horizontal grid lines
    if horizontal_match:
        for i, line in enumerate(horizontal_match):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(horizontal_match)-1 else (255, 255, 0)
            cv2.line(result, (0, int(line[1])), (image.shape[1], int(line[1])), color, 2)

    # Draw grid
    if vertical_match and horizontal_match:
        for i in range(len(vertical_match) - 1):
            for j in range(len(horizontal_match) - 1):
                x1 = int(vertical_match[i][0])
                y1 = int(horizontal_match[j][1])
                x2 = int(vertical_match[i+1][0])
                y2 = int(horizontal_match[j+1][1])
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 1)

    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Grid Match with Orientation')
    plt.axis('off')
    plt.show()

# Main script
if __name__ == "__main__":
    # Get image path from script args
    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    # Detect lines
    vertical_lines, horizontal_lines = detect_grid_lines(image)
    
    # Extend lines
    image_height, image_width = image.shape[:2]
    extended_vertical_lines = [extend_line_vertical(line, image_height) for line in vertical_lines]
    extended_horizontal_lines = [extend_line_horizontal(line, image_width) for line in horizontal_lines]

    # Define the column widths for the vertical grid
    column_widths = [
        292, 129, 83, 224, 125, 240, 190, 170, 87, 81, 87, 211, 126, 257, 85, 126, 215, 121, 254, 205, 125, 205, 215
    ]

    # Define expected row heights (assuming equal spacing)
    row_heights = [image_height // 30] * 29  # 30 rows, 29 spaces between them

    # Find the best match for the vertical grid
    vertical_match, vertical_error = find_best_grid_match(extended_vertical_lines, column_widths, is_vertical=True)

    # Find the best match for the horizontal grid
    horizontal_match, horizontal_error = find_best_grid_match(extended_horizontal_lines, row_heights, is_vertical=False)

    # Visualize results
    visualize_results(image, extended_vertical_lines, extended_horizontal_lines, vertical_match, horizontal_match)

    if vertical_match:
        print(f"Best vertical grid match found with error: {vertical_error}")
    else:
        print("No suitable vertical grid match found")

    if horizontal_match:
        print(f"Best horizontal grid match found with error: {horizontal_error}")
    else:
        print("No suitable horizontal grid match found")