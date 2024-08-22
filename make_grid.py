import cv2
import numpy as np

def detect_and_filter_lines(image_path, min_line_length=100, max_line_gap=100, min_spacing=20, edge_threshold=50):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold to get a binary image
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        print(f"No lines detected in {image_path}")
        return image, []
    
    # Filter lines based on spacing and proximity to edges
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line length and orientation
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        orientation = 'horizontal' if abs(y2 - y1) < abs(x2 - x1) else 'vertical'
        
        # Filter lines near the edges
        if x1 < edge_threshold or x2 < edge_threshold or y1 < edge_threshold or y2 < edge_threshold:
            continue
        
        # Filter based on spacing (eliminate lines that are too close to each other)
        if orientation == 'horizontal':
            if all(abs(y1 - line[0][1]) > min_spacing for line in filtered_lines):
                filtered_lines.append(line)
        else:
            if all(abs(x1 - line[0][0]) > min_spacing for line in filtered_lines):
                filtered_lines.append(line)
    
    print(f"Detected {len(lines)} lines, filtered to {len(filtered_lines)} lines in {image_path}")
    return image, filtered_lines

def process_template(template_path):
    # Load the template image with alpha channel
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        raise ValueError(f"Could not load template image from {template_path}")
    
    # Check if the image has an alpha channel
    if template.shape[2] != 4:
        raise ValueError(f"Template image should have an alpha channel")
    
    # Use the alpha channel as a mask
    alpha = template[:,:,3]
    
    # Threshold the alpha channel to get binary image
    _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to lines
    lines = []
    for contour in contours:
        for i in range(len(contour)):
            x1, y1 = contour[i][0]
            x2, y2 = contour[(i+1) % len(contour)][0]
            lines.append(np.array([[x1, y1, x2, y2]]))
    
    print(f"Detected {len(lines)} line segments in template")
    return template, lines

def align_grid_template(template_path, target_path):
    # Process the template
    template, template_lines = process_template(template_path)
    
    # Detect lines in the target image
    target_image, target_lines = detect_and_filter_lines(target_path)
    
    # Initialize variables for best match
    best_match = None
    best_score = float('-inf')
    
    # Define range for scaling and rotation (can be adjusted based on your needs)
    scaling_factors = np.linspace(0.5, 1.5, 5)
    rotation_angles = range(-30, 31, 10)
    
    for scale in scaling_factors:
        scaled_template_lines = [line * scale for line in template_lines]
        
        for angle in rotation_angles:
            # Rotate the scaled template lines
            rotation_matrix = cv2.getRotationMatrix2D((template.shape[1]//2, template.shape[0]//2), angle, 1)
            rotated_template_lines = []
            for line in scaled_template_lines:
                x1, y1, x2, y2 = line[0]
                pt1 = np.dot(rotation_matrix, [x1, y1, 1])[:2]
                pt2 = np.dot(rotation_matrix, [x2, y2, 1])[:2]
                rotated_template_lines.append(np.array([[pt1[0], pt1[1], pt2[0], pt2[1]]], dtype=np.int32))
            
            # Compare lines in template with lines in target image
            score = 0
            for tline in rotated_template_lines:
                tx1, ty1, tx2, ty2 = tline[0]
                for line in target_lines:
                    x1, y1, x2, y2 = line[0]
                    # Calculate similarity (you might want to use a more sophisticated metric)
                    similarity = abs(tx1 - x1) + abs(ty1 - y1) + abs(tx2 - x2) + abs(ty2 - y2)
                    score += 1 / (1 + similarity)  # Higher score for more similar lines
            
            if score > best_score:
                best_score = score
                best_match = (scale, angle)
    
    if best_match is None:
        print("Could not find a good match for the template")
        return target_image
    
    # Apply best transformation to template
    best_scale, best_angle = best_match
    best_template = cv2.resize(template, None, fx=best_scale, fy=best_scale)
    M = cv2.getRotationMatrix2D((best_template.shape[1] // 2, best_template.shape[0] // 2), best_angle, 1.0)
    aligned_template = cv2.warpAffine(best_template, M, (target_image.shape[1], target_image.shape[0]))
    
    # Overlay aligned template on target image
    overlay = target_image.copy()
    mask = aligned_template[:,:,3]
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(overlay, overlay, mask=mask_inv)
    img2_fg = cv2.bitwise_and(aligned_template[:,:,:3], aligned_template[:,:,:3], mask=mask)
    overlay = cv2.add(img1_bg, img2_fg)
    
    print(f"Best match found at scale {best_scale:.2f} and angle {best_angle}")
    return overlay

# Main execution
if __name__ == "__main__":
    template_path = '/Users/chim/Working/Thesis/Attack_Images/graph_template.png'
    target_path = '/Users/chim/Working/Thesis/Attack_Images/BOX_7/BOOK_14_SUPPLY/IMG_7200.JPG'
    result = align_grid_template(template_path, target_path)
    cv2.imshow('Aligned Grid Overlay', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()