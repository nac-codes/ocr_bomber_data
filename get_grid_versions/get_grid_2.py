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

def extend_line_vertical(line, image_height):
    x1, y1, x2, y2 = line
    # logging.info(f"Extending line {line}")
    if y2 != y1:
        slope = (x2 - x1) / (y2 - y1)
        # Ensure y1 is always the top point and y2 is the bottom point
        if y1 > y2:
            y1, y2 = y2, y1
            x1, x2 = x2, x1
        
        # Extend to top of image (y = 0)
        x1_full = x1 - slope * y1
        
        # Extend to bottom of image (y = image_height)
        x2_full = x1 + slope * (image_height - y1)

        # logging.info(f"Extended line to [{x1_full}, 0, {x2_full}, {image_height}]")
        
        return [x1_full, 0, x2_full, image_height]
    else:
        # If the line is perfectly vertical, just extend it to the full height
        logging.info(f"Extended line to [{x1}, 0, {x2}, {image_height}]")
        return [x1, 0, x1, image_height]
    
def clean_parallel_lines(lines, target_count: int = 28, minimum: int = 24, initial_threshold: float = 2.5, threshold_step: float = 0.1, max_iterations: int = 10):
    def calculate_slope(line):
        return math.atan2(line[3] - line[1], line[2] - line[0])

    def calculate_mad(slopes, median):
        return np.median([abs(slope - median) for slope in slopes])

    cleaned_lines = lines.copy()
    threshold = initial_threshold
    iterations = 1

    logger.info(f"Cleaning parallel lines with initial threshold {threshold}")
    logger.info(f"Initial line count: {len(cleaned_lines)}")

    for _ in range(max_iterations):
        slopes = [calculate_slope(line) for line in cleaned_lines]
        median_slope = np.median(slopes)
        mad = calculate_mad(slopes, median_slope)

        new_cleaned_lines = [
            line for line, slope in zip(cleaned_lines, slopes)
            if abs(slope - median_slope) <= threshold * mad
        ]

        if len(new_cleaned_lines) < minimum:
            logger.info(f"Too few lines found ({len(new_cleaned_lines)}), widening threshold")
            threshold += threshold_step
            iterations += 1
            continue
        else:
            if len(new_cleaned_lines) <= target_count:
                logger.info(f"Returning to {len(new_cleaned_lines)} after {iterations} iterations")
                return new_cleaned_lines

            cleaned_lines = new_cleaned_lines
            threshold -= threshold_step
            iterations += 1
    
    logger.info(f"Failed to converge to {target_count} after {iterations} iterations")
    logger.info(f"Returning {len(cleaned_lines)} lines")

    return cleaned_lines


def clean_vertical_lines(vertical_lines, column_widths, image_height):
    threshold = 60
    cleaned_lines = []

    margin_bottom = 5
    margin_top = 6

    while (len(cleaned_lines) < len(column_widths) + margin_bottom or len(cleaned_lines) > len(column_widths) + margin_top) and 35 < threshold < 100:
        cleaned_lines = []
        logger.info(f"Cleaning vertical lines with threshold {threshold}")
        
        for line in vertical_lines:
            extended_line = extend_line_vertical(line, image_height)
            x1, y1, x2, y2 = extended_line
            to_add = True
            
            for i, cleaned_line in enumerate(cleaned_lines):
                x3, y3, x4, y4 = cleaned_line            
                
                
                if (abs(x1 - x3) < threshold and abs(x2 - x4) < threshold):
                    # Merge the lines by averaging their positions
                    new_x1 = (x1 + x3) / 2
                    new_x2 = (x2 + x4) / 2
                    cleaned_lines[i] = [new_x1, 0, new_x2, image_height]
                    to_add = False
                    break
            
            if to_add:
                cleaned_lines.append(extended_line)

        if len(cleaned_lines) < len(column_widths) + margin_bottom:
            logger.info(f"Too few lines found ({len(cleaned_lines)}), decreasing threshold")
            threshold -= 5
        elif len(cleaned_lines) > len(column_widths) + margin_top:
            logger.info(f"Too many lines found ({len(cleaned_lines)}), increasing threshold")
            threshold += 5

    cleaned_lines = clean_parallel_lines(cleaned_lines, len(column_widths) + 4, len(column_widths) + 1)

    # Sort the cleaned lines by their midpoint x-coordinate
    cleaned_lines.sort(key=lambda line: (line[0] + line[2]) / 2)

    return cleaned_lines

def extend_line_horizontal(line, image_width):
    x1, y1, x2, y2 = line
    # logging.info(f"Extending line {line}")
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        # Ensure x1 is always the left point and x2 is the right point
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        # Extend to left of image (x = 0)
        y1_full = y1 - slope * x1
        
        # Extend to right of image (x = image_width)
        y2_full = y1 + slope * (image_width - x1)

        # logging.info(f"Extended line to [0, {y1_full}, {image_width}, {y2_full}]")
        
        return [0, y1_full, image_width, y2_full]
    else:
        # If the line is perfectly horizontal, just extend it to the full width
        logging.info(f"Extended line to [0, {y1}, {image_width}, {y2}]")
        return [0, y1, image_width, y1]


def clean_horizontal_lines(horizontal_lines, rows, image_width):
    threshold = 60
    cleaned_lines = []

    margin_bottom = 4
    margin_top = 6

    while (len(cleaned_lines) < rows + margin_bottom or len(cleaned_lines) > rows + margin_top) and 35 < threshold < 100:
        cleaned_lines = []
        logger.info(f"Cleaning horizontal lines with threshold {threshold}")
        
        for line in horizontal_lines:
            extended_line = extend_line_horizontal(line, image_width)
            x1, y1, x2, y2 = extended_line
            to_add = True
            
            for i, cleaned_line in enumerate(cleaned_lines):
                x3, y3, x4, y4 = cleaned_line
                
                if (abs(y1 - y3) < threshold and abs(y2 - y4) < threshold):
                    # Merge the lines by averaging their positions
                    new_y1 = (y1 + y3) / 2
                    new_y2 = (y2 + y4) / 2
                    cleaned_lines[i] = [0, new_y1, image_width, new_y2]
                    to_add = False
                    break
            
            if to_add:
                cleaned_lines.append(extended_line)

        if len(cleaned_lines) < rows + margin_bottom:
            logger.info(f"Too few lines found ({len(cleaned_lines)}), decreasing threshold")
            threshold -= 5
        elif len(cleaned_lines) > rows + margin_top:
            logger.info(f"Too many lines found ({len(cleaned_lines)}), increasing threshold")
            threshold += 5

    cleaned_lines = clean_parallel_lines(cleaned_lines, rows + 5, rows + 1)

    # Sort the cleaned lines by their midpoint y-coordinate
    cleaned_lines.sort(key=lambda line: (line[1] + line[3]) / 2)

    return cleaned_lines

def find_best_vertical_grid_match(vertical_lines, column_widths):
    best_match = None
    best_error = float('inf')
    total_width = sum(column_widths)
    normalized_widths = [width / total_width for width in column_widths]

    lines_to_find = len(column_widths) + 1

    logger.info(f"Total vertical lines: {len(vertical_lines)}")
    logger.info(f"Lines to find: {lines_to_find}")

    # check if the the difference between number of vertical lines and lines to find is greater than 10 and if so, return None
    if abs(len(vertical_lines) - lines_to_find) > 10:
        logger.info(f"Too many vertical lines to find a match")
        return None, None, vertical_lines

    combos = itertools.combinations(range(len(vertical_lines)), lines_to_find)
    total_combos = sum(1 for _ in combos)
    logger.info(f"Total combos: {total_combos}")

    # process all combos in combos
    for combo in tqdm(combos,
                        total=total_combos,
                        desc="Processing combinations"):
        segment_lines = [vertical_lines[i] for i in combo]
        
        # Calculate widths between lines
        segment_widths = [segment_lines[i+1][0] - segment_lines[i][0] for i in range(len(segment_lines)-1)]
        total_segment_width = segment_lines[-1][0] - segment_lines[0][0]
        
        if total_segment_width == 0:
            continue
        
        normalized_segments = [width / total_segment_width for width in segment_widths]
        error = sum((a - b) ** 2 for a, b in zip(normalized_segments, normalized_widths))
        
        # logger.info(f"Checking combination {combo}, error: {error}")
        
        if error < best_error:
            best_error = error
            best_match = combo
            logger.info(f"-----New best match found: {best_match}")

    return best_match, best_error


def find_best_horizontal_grid_match(horizontal_lines, expected_count=30, expected_distance=110):
    best_match = None
    best_error = float('inf')

    logger.info(f"Total horizontal lines: {len(horizontal_lines)}")
    logger.info(f"Lines to find: {expected_count}")

    if abs(len(horizontal_lines) - expected_count) > 10:
        logger.info(f"Too many horizontal lines to find a match")
        return None, None

    # Generate all possible combinations of expected_count lines
    for combo in tqdm(itertools.combinations(range(len(horizontal_lines)), expected_count), 
                      total=sum(1 for _ in itertools.combinations(range(len(horizontal_lines)), expected_count)), 
                      desc="Processing combinations"):
        segment = [horizontal_lines[i] for i in combo]
        
        distances = [segment[i+1][1] - segment[i][1] for i in range(len(segment) - 1)]
        error = sum((d - expected_distance) ** 2 for d in distances) / len(distances)

        if error < best_error:
            best_error = error
            best_match = combo
            logger.info(f"New best horizontal match found: {best_match}, error: {error}")

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

def is_more_vertical(line):
    x1, y1, x2, y2 = line
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    return dy > dx

# Main script
if __name__ == "__main__":
    # Get image path from script args
    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    # Detect lines
    vertical_lines, horizontal_lines = detect_grid_lines(image)
    
    #save vertical and horizontal lines as text file
    with open("vertical_lines.txt", "w") as f:
        for line in vertical_lines:
            f.write(str(line) + "\n")
    with open("horizontal_lines.txt", "w") as f:
        for line in horizontal_lines:
            f.write(str(line) + "\n")
        
    

    # visualize the vertical and horizontal lines
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    # Define the column widths for the vertical grid
    column_widths = [
        292,  # First column (Day)
        129, 83, 224, 125, 240, 190, 170, 87, 81, 87, 211, 126, 257, 85, 126, 215, 121, 254, 205, 125, 205,
        215  # Last column (Total Tons)
    ]

    cleaned_vertical_lines = clean_vertical_lines(vertical_lines, column_widths, image.shape[0])
    cleaned_horizontal_lines = clean_horizontal_lines(horizontal_lines, 30, image.shape[1])

    # print out number of lines of each
    logger.info(f"Cleaned vertical lines: {len(cleaned_vertical_lines)}")
    logger.info(f"Cleaned horizontal lines: {len(cleaned_horizontal_lines)}")

    # clear the image
    image = cv2.imread(image_path)

    # Visualize the cleaned vertical and horizontal lines
    for x1, y1, x2, y2 in cleaned_vertical_lines:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
    for x1, y1, x2, y2 in cleaned_horizontal_lines:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    
    cv2.imshow("image", image)
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