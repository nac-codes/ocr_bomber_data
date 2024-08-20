import os
import argparse
from PIL import Image
import pyheif

def rotate_image(image_path):
    try:
        # Open the image file
        img = Image.open(image_path)
        
        # Rotate the image by 180 degrees
        rotated_img = img.rotate(180)
        
        # Save the rotated image back to the same path
        rotated_img.save(image_path)
        print(f"Rotated {image_path}")
        
    except Exception as e:
        print(f"Error rotating {image_path}: {e}")

def rotate_cr3_image(cr3_path):
    try:
        # Open the CR3 file using pyheif
        heif_file = pyheif.read(cr3_path)
        
        # Convert the HEIF file to a PIL image
        img = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        
        # Rotate the image by 180 degrees
        rotated_img = img.rotate(180)
        
        # Save the rotated image back to the same path
        rotated_img.save(cr3_path.replace('.CR3', '.jpg'))
        print(f"Rotated {cr3_path} and saved as JPEG")
        
    except Exception as e:
        print(f"Error rotating {cr3_path}: {e}")

def rotate_images_in_directory(directory):
    # Walk through all directories and files
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                rotate_image(os.path.join(root, filename))
            elif filename.lower().endswith('.cr3'):
                rotate_cr3_image(os.path.join(root, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rotate all JPEG and CR3 images in a directory (and its subdirectories) by 180 degrees.')
    parser.add_argument('directory', type=str, help='The path to the directory containing the images')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"The provided directory does not exist: {args.directory}")
    else:
        rotate_images_in_directory(args.directory)
