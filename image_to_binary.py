import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import binarize
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Constants
IMAGE_SIZE = (64, 64)  # Target size for the pattern
default_input_dir = "26alphabetwords"
default_output_dir = "patterns_from_images"

def preprocess_image(image_path, size=IMAGE_SIZE):
    """Load, resize, binarize and flatten an image."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(size, Image.NEAREST)
    img_array = np.array(img)
    binary_img = binarize(img_array.reshape(1, -1), threshold=127).reshape(size)  # Reshape to 2D
    return binary_img.astype(int)

def convert_images_to_patterns(input_dir=default_input_dir, output_dir=default_output_dir):
    """Converts images in the input_images folder to patterns and saves them."""
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}' for storing patterns.")

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No image files found in '{input_dir}'.")
        return

    print("Converting images to patterns:")
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        name = os.path.splitext(image_file)[0]
        pattern_file = os.path.join(output_dir, f"{name}.txt")
        if not os.path.exists(pattern_file):
            try:
                binary_matrix = preprocess_image(image_path)
                np.savetxt(pattern_file, binary_matrix, fmt='%d')
                print(f"  Converted '{image_file}' to pattern '{name}.txt'")
            except Exception as e:
                print(f"  Error converting '{image_file}': {e}")
        else:
            print(f"  Pattern '{name}.txt' already exists. Skipping conversion.")

if __name__ == "__main__":
    convert_images_to_patterns()
