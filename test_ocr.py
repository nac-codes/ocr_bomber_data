import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import argparse

def extract_text_with_hocr(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Run Tesseract OCR with HOCR output
    hocr = pytesseract.image_to_pdf_or_hocr(image, extension='hocr')

    # Save the HOCR output to a file (optional)
    with open("output.hocr", "wb") as hocr_file:
        hocr_file.write(hocr)

    # Extract OCR data with bounding boxes
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)

    # Draw bounding boxes around detected words
    for i in range(len(ocr_data['text'])):
        if int(ocr_data['conf'][i]) > 0:  # Filter out low-confidence results
            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], 
                          ocr_data['width'][i], ocr_data['height'][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the image from OpenCV format to PIL format to display
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_pil.show()

    # Save the image with bounding boxes (optional)
    output_image_path = "output_with_boxes.jpg"
    image_pil.save(output_image_path)
    print(f"HOCR output saved as 'output.hocr' and image with bounding boxes saved as '{output_image_path}'.")

# get file path with argparse
parser = argparse.ArgumentParser(description='Extract text from an image using Tesseract OCR with HOCR output.')
parser.add_argument('image_path', type=str, help='The path to the image file')

args = parser.parse_args()
image_path = args.image_path

# Call the function with the specified image path
extract_text_with_hocr(image_path)
