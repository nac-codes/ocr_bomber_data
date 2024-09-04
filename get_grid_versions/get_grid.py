import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import logging
import sys

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

    # Group lines by orientation (vertical or horizontal)
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):
            horizontal_lines.append(line[0])
        else:
            vertical_lines.append(line[0])
    
    if visualize:
        # Visualize initial detected lines
        initial_lines_image = image.copy()
        for line in horizontal_lines + vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(initial_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(initial_lines_image, cv2.COLOR_BGR2RGB))
        plt.title('Initially Detected Lines')
        plt.axis('off')
    
    # Function to merge nearby parallel lines
    def merge_lines(lines, is_horizontal):
        if not lines:
            return []
        lines = sorted(lines, key=lambda x: x[1] if is_horizontal else x[0])
        merged = [lines[0]]
        for line in lines[1:]:
            prev = merged[-1]
            if is_horizontal:
                if abs(line[1] - prev[1]) < 10:  # Adjust threshold as needed
                    merged[-1] = (min(prev[0], line[0]), (prev[1] + line[1]) // 2, 
                                  max(prev[2], line[2]), (prev[3] + line[3]) // 2)
                else:
                    merged.append(line)
            else:
                if abs(line[0] - prev[0]) < 10:  # Adjust threshold as needed
                    merged[-1] = ((prev[0] + line[0]) // 2, min(prev[1], line[1]),
                                  (prev[2] + line[2]) // 2, max(prev[3], line[3]))
                else:
                    merged.append(line)
        return merged

    merged_horizontal = merge_lines(horizontal_lines, True)
    merged_vertical = merge_lines(vertical_lines, False)
    
    if visualize:
        # Visualize merged lines
        merged_lines_image = image.copy()
        for line in merged_horizontal + merged_vertical:
            x1, y1, x2, y2 = line
            cv2.line(merged_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(merged_lines_image, cv2.COLOR_BGR2RGB))
        plt.title('Merged Lines')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    logger.info(f"Detected {len(merged_horizontal)} horizontal and {len(merged_vertical)} vertical lines in the target image")
    return merged_horizontal + merged_vertical

def find_best_grid_match(vertical_lines, column_widths):
    best_match = None
    best_error = float('inf')
    total_width = sum(column_widths)
    normalized_widths = [width / total_width for width in column_widths]

    logger.info(f"-------------- {len(vertical_lines)} vertical lines before cleaning")

    # clean up vertical lines, extend them to span the whole height of the image and remove lines within a certain x distance of each other
    cleaned_lines = []
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        to_add = True
        for line2 in cleaned_lines:
            x3, y3, x4, y4 = line2
            if abs(x1 - x3) < 50:
                # adjust the values of the matching cleaned line to create an average of x1 and x3 and x2 and x4
                line2[0] = (x1 + x3) / 2
                line2[2] = (x2 + x4) / 2
                cleaned_lines[cleaned_lines.index(line2)] = line2
                to_add = False
                break

        if to_add:
            # logger.info(f"Adding line {line}")
            cleaned_lines.append([x1, 0, x2, 4000])


        
    vertical_lines = cleaned_lines

    logger.info(f"--------------Found {len(vertical_lines)} vertical lines")

    # display all vertical lines on cv2 overalyed on image in yellow
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Grid Match')
    plt.axis('off')
    plt.show()



    logger.info(f"Columns widths: {column_widths}")
    lines_to_find = len(column_widths) + 2

    logger.info(f"Searching in range {len(vertical_lines) - lines_to_find + 1}")

    for start in range(len(vertical_lines) - lines_to_find + 1):
        logger.info(f"--Checking segment starting at line {start}")
        # get the first and last line in the segment
        start_line = vertical_lines[start]
        end_line = vertical_lines[start + lines_to_find - 1]

        # get the width between each line in the segment
        prev_line = start_line
        segment_widths = []
        for i in range(1, lines_to_find):
            line = vertical_lines[start + i]
            width = line[0] - prev_line[0]
            segment_widths.append(width)
            prev_line = line

        # get the total segment width by taking the x coordinate of the last line in the segment and subtracting the x coordinate of the first line in the segment
        total_segment_width = end_line[0] - start_line[0]
        
        
        if total_segment_width == 0:
            continue
        
        normalized_segments = [width / total_segment_width for width in segment_widths]
        error = sum((a - b) ** 2 for a, b in zip(normalized_segments, normalized_widths))
        logger.info(f"Segment error: {error}")
        
        if error < best_error:
            best_error = error
            best_match = (start, start + i - 1)
            logger.info(f"-----New best match found: {best_match}")
    
    return best_match, best_error, vertical_lines

def visualize_results(image, vertical_lines, grid_match):
    result = image.copy()
    
    # Draw all vertical lines
    # for x1, y1, x2, y2 in vertical_lines:
    #     cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    
    # Draw matched grid lines, start line draw green, end line draw red
    if grid_match:
        start, end = grid_match
        cv2.line(result, (int(vertical_lines[start][0]), int(vertical_lines[start][1])),
                 (int(vertical_lines[start][2]), int(vertical_lines[start][3])), (0, 255, 0), 2)
        cv2.line(result, (int(vertical_lines[end][0]), int(vertical_lines[end][1])),
                 (int(vertical_lines[end][2]), int(vertical_lines[end][3])), (0, 0, 255), 2)
        # all inbetween lines draw blue
        for i in range(start + 1, end):
            cv2.line(result, (int(vertical_lines[i][0]), int(vertical_lines[i][1])),
                     (int(vertical_lines[i][2]), int(vertical_lines[i][3])), (255, 0, 0), 2)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Grid Match')
    plt.axis('off')
    plt.show()

# Main script
if __name__ == "__main__":
    # Get image path from script args
    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    # Detect lines
    all_lines = detect_grid_lines(image)
    
    # Separate vertical and horizontal lines
    vertical_lines = [line for line in all_lines if abs(line[2] - line[0]) < abs(line[3] - line[1])]
    
    # Sort vertical lines by x-coordinate
    vertical_lines.sort(key=lambda line: line[0])

    # Define the column widths for the entire grid
    column_widths = [
        129, 83,  # Month, Year 292, 129, 83, -- originally day month year removed the first column because it's sometimes not present
        224,  # Time of Attack
        125,  # Air Force
        240,  # Group/Squadron Number
        190,  # Number of Aircraft Bombing
        170,  # Altitude of Release in Hund. Ft.
        87,   # Sighting
        81,   # Visibility of Target
        87,   # Target Priority
        211, 126, 257, 85, 126,  # High Explosive Bombs
        215, 121, 254,  # Incendiary Bombs
        205, 125, 205  # Fragmentation Bombs
        # 215   # Total Tons This column is also oftentimes not present
    ]

    # Find the best match for the entire grid
    best_match, error, vertical_lines = find_best_grid_match(vertical_lines, column_widths)

    # Visualize results
    visualize_results(image, vertical_lines, best_match)

    if best_match:
        start, end = best_match
        print(f"Best grid match found from line {start} to {end}")
        print(f"Match error: {error}")
    else:
        print("No suitable grid match found")