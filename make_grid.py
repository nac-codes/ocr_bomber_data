import cv2
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_template(template_path):
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        raise ValueError(f"Could not load template image from {template_path}")
    
    if template.shape[2] != 4:
        raise ValueError(f"Template image should have an alpha channel")
    
    alpha = template[:,:,3]
    _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    
    # Use probabilistic Hough transform to detect line segments
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        raise ValueError("No lines detected in the template")
    
    # Merge nearby parallel lines
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
            y = (y1 + y2) // 2
            merged_lines.append([0, y, binary.shape[1], y])
        else:  # Vertical line
            x = (x1 + x2) // 2
            merged_lines.append([x, 0, x, binary.shape[0]])
    
    logging.info(f"Detected {len(merged_lines)} lines in template")
    return template, np.array(merged_lines)

def detect_and_filter_lines(image_path, min_line_length=1000, max_line_gap=100):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        logging.warning("No lines detected in the target image")
        return image, []
    
    logging.info(f"Detected {len(lines)} lines in target image")
    return image, lines

def calculate_score(template_lines, target_lines):
    score = 0
    for tline in template_lines:
        for line in target_lines:
            if do_lines_intersect(tline, line[0]):
                score += 1
    return score

def align_grid_template(template_path, target_path):
    template, template_lines = process_template(template_path)
    target_image, target_lines = detect_and_filter_lines(target_path)
    
    # Coarse search parameters
    coarse_scaling_factors = np.linspace(0.9, 1.1, 9)
    coarse_rotation_angles = range(-10, 10, 5)
    coarse_x_offsets = range(-25, 25, 10)
    coarse_y_offsets = range(-25, 25, 10)
    
    # Fine search parameters
    fine_scale_range = 0.05
    fine_angle_range = 1
    fine_offset_range = 2
    fine_steps = 2
    
    scores = []
    params = []
    
    # Coarse search
    total_iterations = len(coarse_scaling_factors) * len(coarse_rotation_angles) * len(coarse_x_offsets) * len(coarse_y_offsets)
    with tqdm(total=total_iterations, desc="Coarse search") as pbar:
        for scale in coarse_scaling_factors:
            scaled_template_lines = template_lines * scale
            for angle in coarse_rotation_angles:
                rotation_matrix = cv2.getRotationMatrix2D((template.shape[1]//2, template.shape[0]//2), angle, 1)
                for x_offset in coarse_x_offsets:
                    for y_offset in coarse_y_offsets:
                        rotated_template_lines = []
                        for line in scaled_template_lines:
                            x1, y1, x2, y2 = line
                            pt1 = np.dot(rotation_matrix, [x1+x_offset, y1+y_offset, 1])[:2]
                            pt2 = np.dot(rotation_matrix, [x2+x_offset, y2+y_offset, 1])[:2]
                            rotated_template_lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])
                        
                        score = calculate_score(rotated_template_lines, target_lines)
                        scores.append(score)
                        params.append((scale, angle, x_offset, y_offset))
                        pbar.update(1)
    
    # Visualize the distribution
    visualize_distribution(scores, params)
    
    # Select top candidates for fine search
    threshold = np.percentile(scores, 98)  # Top 5% of scores
    top_candidates = [param for score, param in zip(scores, params) if score >= threshold]
    
    # Fine search
    best_score = max(scores)
    best_params = params[np.argmax(scores)]
    checked_params = set()
    
    total_iterations = len(top_candidates) * fine_steps**4
    with tqdm(total=total_iterations, desc="Fine search") as pbar:
        for candidate in top_candidates:
            scale, angle, x_offset, y_offset = candidate
            for fine_scale in np.linspace(scale - fine_scale_range, scale + fine_scale_range, fine_steps):
                for fine_angle in range(angle - fine_angle_range, angle + fine_angle_range + 1):
                    for fine_x in range(x_offset - fine_offset_range, x_offset + fine_offset_range + 1, 2):
                        for fine_y in range(y_offset - fine_offset_range, y_offset + fine_offset_range + 1, 2):
                            params = (fine_scale, fine_angle, fine_x, fine_y)
                            if params in checked_params:
                                continue
                            checked_params.add(params)
                            
                            scaled_template_lines = template_lines * fine_scale
                            rotation_matrix = cv2.getRotationMatrix2D((template.shape[1]//2, template.shape[0]//2), fine_angle, 1)
                            rotated_template_lines = []
                            for line in scaled_template_lines:
                                x1, y1, x2, y2 = line
                                pt1 = np.dot(rotation_matrix, [x1+fine_x, y1+fine_y, 1])[:2]
                                pt2 = np.dot(rotation_matrix, [x2+fine_x, y2+fine_y, 1])[:2]
                                rotated_template_lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])
                            
                            score = calculate_score(rotated_template_lines, target_lines)
                            if score > best_score:
                                best_score = score
                                best_params = params
                            pbar.update(1)
    
    logging.info(f"Best match: scale={best_params[0]:.2f}, angle={best_params[1]}, x_offset={best_params[2]}, y_offset={best_params[3]}")
    
    # Apply best transformation to template
    best_scale, best_angle, best_x_offset, best_y_offset = best_params
    M = cv2.getRotationMatrix2D((template.shape[1]//2, template.shape[0]//2), best_angle, best_scale)
    M[0, 2] += best_x_offset
    M[1, 2] += best_y_offset
    aligned_template = cv2.warpAffine(template, M, (target_image.shape[1], target_image.shape[0]))
    
    # Overlay aligned template on target image
    mask = aligned_template[:,:,3]
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(target_image, target_image, mask=mask_inv)
    img2_fg = cv2.bitwise_and(aligned_template[:,:,:3], aligned_template[:,:,:3], mask=mask)
    result = cv2.add(img1_bg, img2_fg)
    
    return result

def visualize_distribution(scores, params):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scales = [p[0] for p in params]
    angles = [p[1] for p in params]
    x_offsets = [p[2] for p in params]
    y_offsets = [p[3] for p in params]
    
    scatter = ax.scatter(scales, angles, scores, c=scores, cmap='viridis')
    ax.set_xlabel('Scale')
    ax.set_ylabel('Angle')
    ax.set_zlabel('Score')
    
    plt.colorbar(scatter, label='Score')
    
    threshold = np.percentile(scores, 98)
    ax.plot([min(scales), max(scales)], [min(angles), max(angles)], [threshold, threshold], 'r--', label='Threshold (98th percentile)')
    
    plt.title('Distribution of Alignment Scores')
    plt.legend()
    plt.savefig('alignment_distribution.png')
    plt.close()

def do_lines_intersect(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calculate the denominator
    den = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if den == 0:
        return False
    
    # Calculate the intersection point
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / den
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / den
    
    # Check if the intersection point lies on both line segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return True
    return False

# Main execution
if __name__ == "__main__":
    template_path = '/Users/chim/Working/Thesis/Attack_Images/OCR/graph_template.png'
    target_path = '/Users/chim/Working/Thesis/Attack_Images/BOX_7/BOOK_14_SUPPLY/IMG_7200.JPG'
    result = align_grid_template(template_path, target_path)
    cv2.imshow('Aligned Grid Overlay', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()