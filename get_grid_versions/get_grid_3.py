import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import itertools
from tqdm import tqdm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from math import atan2, degrees, cos, sin, radians
import math
from scipy.stats import linregress

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

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from tqdm import tqdm
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def clean_vertical_lines(vertical_lines, image_height):
    cleaned_lines = []
    threshold = 5

    for line in vertical_lines:
        extended_line = extend_line_vertical(line, image_height)
        x1, _, x2, _ = extended_line
        to_add = True

        for i, (cx1, _, cx2, _) in enumerate(cleaned_lines):
            if abs(x1 - cx1) < threshold and abs(x2 - cx2) < threshold:
                cleaned_lines[i] = [(x1 + cx1) / 2, 0, (x2 + cx2) / 2, image_height]
                to_add = False
                break

        if to_add:
            cleaned_lines.append(extended_line)

    return sorted(cleaned_lines, key=lambda line: line[0])

def clean_horizontal_lines(horizontal_lines, image_width):
    cleaned_lines = []
    threshold =   # 10 pixels threshold

    for line in horizontal_lines:
        extended_line = extend_line_horizontal(line, image_width)
        _, y1, _, y2 = extended_line
        to_add = True

        for i, (_, cy1, _, cy2) in enumerate(cleaned_lines):
            if abs(y1 - cy1) < threshold and abs(y2 - cy2) < threshold:
                cleaned_lines[i] = [0, (y1 + cy1) / 2, image_width, (y2 + cy2) / 2]
                to_add = False
                break

        if to_add:
            cleaned_lines.append(extended_line)

    return sorted(cleaned_lines, key=lambda line: line[1])

#REDO THIS FUNCTION
def find_best_vertical_grid_match(vertical_lines, column_widths):
    total_width = sum(column_widths)
    normalized_widths = [width / total_width for width in column_widths]
    lines_to_find = len(column_widths) + 1

    logger.info(f"Total vertical lines: {len(vertical_lines)}")
    logger.info(f"Lines to find: {lines_to_find}")

    def calculate_error(segment_lines):
        segment_widths = [segment_lines[i+1][0] - segment_lines[i][0] for i in range(len(segment_lines)-1)]
        total_segment_width = segment_lines[-1][0] - segment_lines[0][0]
        
        if total_segment_width == 0:
            return float('inf')
        
        normalized_segments = [width / total_segment_width for width in segment_widths]
        return sum((a - b) ** 2 for a, b in zip(normalized_segments, normalized_widths))

    best_match = None
    best_error = float('inf')

    # Start with a full set of lines and gradually remove lines
    current_lines = list(range(len(vertical_lines)))

    # init progress bar
    pbar = tqdm(total=len(current_lines) - lines_to_find)
    
    while len(current_lines) >= lines_to_find:
        error = calculate_error([vertical_lines[i] for i in current_lines])
        
        if error < best_error:
            best_error = error
            best_match = current_lines
            # logger.info(f"New best match found: {best_match}, error: {error}")
        
        if len(current_lines) == lines_to_find:
            break
        
        # Remove the line that results in the lowest error when removed
        min_error = float('inf')
        line_to_remove = None
        
        for i in range(len(current_lines)):
            test_lines = current_lines[:i] + current_lines[i+1:]
            error = calculate_error([vertical_lines[j] for j in test_lines])
            
            if error < min_error:
                min_error = error
                line_to_remove = i
        
        current_lines.pop(line_to_remove)
        
        pbar.update(1)

    return best_match, best_error

#REDO THIS FUNCTION
def find_best_horizontal_grid_match(horizontal_lines, expected_count=30, expected_distance=110):
    logger.info(f"Total horizontal lines: {len(horizontal_lines)}")
    logger.info(f"Lines to find: {expected_count}")

    def calculate_error(segment):
        distances = [segment[i+1][1] - segment[i][1] for i in range(len(segment) - 1)]
        return sum((d - expected_distance) ** 2 for d in distances) / len(distances)

    best_match = None
    best_error = float('inf')

    # Start with a full set of lines and gradually remove lines
    current_lines = list(range(len(horizontal_lines)))

    #init progress bar
    pbar = tqdm(total=len(current_lines) - expected_count)
    
    while len(current_lines) >= expected_count:
        error = calculate_error([horizontal_lines[i] for i in current_lines])
        
        if error < best_error:
            best_error = error
            best_match = current_lines
            # logger.info(f"New best horizontal match found: {best_match}, error: {error}")
        
        if len(current_lines) == expected_count:
            break
        
        # Remove the line that results in the lowest error when removed
        min_error = float('inf')
        line_to_remove = None
        
        for i in range(len(current_lines)):
            test_lines = current_lines[:i] + current_lines[i+1:]
            error = calculate_error([horizontal_lines[j] for j in test_lines])
            
            if error < min_error:
                min_error = error
                line_to_remove = i
        
        current_lines.pop(line_to_remove)

        #update progress bar
        pbar.update(1)

    return best_match, best_error

def visualize_results(image, vertical_lines, horizontal_lines, vertical_match, horizontal_match):
    result = image.copy()
    
    # Draw matched vertical grid lines
    if vertical_match:
        for i, index in enumerate(vertical_match):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(vertical_match)-1 else (255, 0, 0)
            line = vertical_lines[index]
            cv2.line(result, (int(line[0]), 0), (int(line[0]), image.shape[0]), color, 2)

    # Draw matched horizontal grid lines
    if horizontal_match:
        for i, index in enumerate(horizontal_match):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == len(horizontal_match)-1 else (255, 255, 0)
            line = horizontal_lines[index]
            cv2.line(result, (0, int(line[1])), (image.shape[1], int(line[1])), color, 2)

    # Draw grid
    if vertical_match and horizontal_match:
        for i in range(len(vertical_match) - 1):
            for j in range(len(horizontal_match) - 1):
                x1 = int(vertical_lines[vertical_match[i]][0])
                y1 = int(horizontal_lines[horizontal_match[j]][1])
                x2 = int(vertical_lines[vertical_match[i+1]][0])
                y2 = int(horizontal_lines[horizontal_match[j+1]][1])
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
    
    # Save vertical and horizontal lines as text file
    with open("vertical_lines.txt", "w") as f:
        for line in vertical_lines:
            f.write(str(line) + "\n")
    with open("horizontal_lines.txt", "w") as f:
        for line in horizontal_lines:
            f.write(str(line) + "\n")

    # Visualize the detected vertical and horizontal lines
    debug_image = image.copy()
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    
    cv2.imshow("Detected Lines", debug_image)
    cv2.waitKey(0)
    
    # Define the column widths for the vertical grid
    column_widths = [
        292,  # First column (Day)
        129, 83, 224, 125, 240, 190, 170, 87, 81, 87, 211, 126, 257, 85, 126, 215, 121, 254, 205, 125, 205,
        215  # Last column (Total Tons)
    ]

    cleaned_vertical_lines = clean_vertical_lines(vertical_lines, image.shape[0])
    cleaned_horizontal_lines = clean_horizontal_lines(horizontal_lines, image.shape[1])

    # Print out number of lines of each
    logger.info(f"Cleaned vertical lines: {len(cleaned_vertical_lines)}")
    logger.info(f"Cleaned horizontal lines: {len(cleaned_horizontal_lines)}")

    # Visualize the cleaned vertical and horizontal lines
    cleaned_image = image.copy()
    for x1, y1, x2, y2 in cleaned_vertical_lines:
        cv2.line(cleaned_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
    for x1, y1, x2, y2 in cleaned_horizontal_lines:
        cv2.line(cleaned_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    
    cv2.imshow("Cleaned Lines", cleaned_image)
    cv2.waitKey(0)

    # Find the best match for the vertical grid
    vertical_match, vertical_error = find_best_vertical_grid_match(cleaned_vertical_lines, column_widths)

    # Find the best match for the horizontal grid
    horizontal_match, horizontal_error = find_best_horizontal_grid_match(cleaned_horizontal_lines)

    # Visualize results
    visualize_results(image, cleaned_vertical_lines, cleaned_horizontal_lines, vertical_match, horizontal_match)

    if vertical_match:
        print(f"Best vertical grid match {vertical_match}")
        print(f"Vertical match error: {vertical_error}")
    else:
        print("No suitable vertical grid match found")

    if horizontal_match:
        print(f"Best horizontal grid match {horizontal_match}")
        print(f"Horizontal match error: {horizontal_error}")
    else:
        print("No suitable horizontal grid match found")