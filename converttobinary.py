import numpy as np
from skimage import io, color, transform

# Constants
IMAGE_SIZE = (8, 8)  # Target size for the image (8x8)
PATTERN_STORE = {}   # Dictionary to store patterns with custom names

def preprocess_image(image_path):
    """
    Load an image, resize to 8x8, and convert to grayscale.
    """
    try:
        image = io.imread(image_path)
        image_resized = transform.resize(image, IMAGE_SIZE, anti_aliasing=True)
        image_gray = color.rgb2gray(image_resized)
        return image_gray
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


def convert_to_binary(image):
    """
    Convert a grayscale image to a binary matrix based on a threshold.
    """
    threshold = np.mean(image)  # Compute the threshold
    binary_matrix = (image > threshold).astype(int)  # Convert to binary
    return binary_matrix

def display_binary_grid(binary_matrix):
    """
    Display the binary matrix in an 8x8 grid.
    """
    for row in binary_matrix:
        print(" ".join(map(str, row)))

def compare_images(binary_matrix1, binary_matrix2):
    """
    Compare two binary matrices and compute similarity metrics.
    Metrics:
    - Hamming Distance: Proportion of differing bits.
    """
    if binary_matrix1.shape != binary_matrix2.shape:
        raise ValueError("Binary matrices must have the same dimensions for comparison.")

    # Flatten matrices for comparison
    flat1 = binary_matrix1.flatten()
    flat2 = binary_matrix2.flatten()

    # Compute Hamming Distance
    hamming_distance = np.sum(flat1 != flat2) / len(flat1)
    return hamming_distance

def store_pattern(image_path, name):
    """
    Preprocess an image, convert it to binary, and store it in the pattern dictionary with a custom name.
    """
    try:
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)
        PATTERN_STORE[name] = binary_matrix
        print(f"Pattern '{name}' stored successfully.")
    except Exception as e:
        print(f"Error storing pattern: {e}")

def recognize_pattern(image_path):
    """
    Recognize a pattern by comparing the input image with stored patterns.
    """
    try:
        # Preprocess the input image
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)

        # Compare with stored patterns
        similarities = {}
        for name, pattern in PATTERN_STORE.items():
            hamming_distance = compare_images(binary_matrix, pattern)
            similarities[name] = hamming_distance

        # Find the closest match
        if similarities:
            best_match = min(similarities, key=similarities.get)
            best_distance = similarities[best_match]
            print(f"Best match: '{best_match}' with Hamming Distance: {best_distance:.2%}")
        else:
            print("No patterns stored for recognition.")
    except Exception as e:
        print(f"Error recognizing pattern: {e}")

def main():
    while True:
        print("\nMenu:")
        print("1. Store a pattern")
        print("2. Recognize a pattern")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            image_path = input("Enter the path to the image file: ").strip()
            name = input("Enter a custom name for this pattern: ").strip()
            store_pattern(image_path, name)
        elif choice == "2":
            image_path = input("Enter the path to the image file for recognition: ").strip()
            recognize_pattern(image_path)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()