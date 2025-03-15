import numpy as np
from skimage import io, color, transform
import os
import warnings
from PIL import Image

# Suppress the specific UserWarning about palette images
warnings.filterwarnings("ignore", category=UserWarning, message="Palette images with Transparency.*")

# Constants
IMAGE_SIZE = (8, 8)  # Target size for the image (8x8)
PATTERN_STORE = {}   # Dictionary to store patterns with custom names
PATTERNS_DIR = "patterns"  # Directory to store pattern text files

class AssociativeMemory:
    def __init__(self, N):
        """
        Initialize the associative memory with size N.
        """
        self.N = N  # Size of the patterns
        self.W = np.zeros((N, N))  # Weight matrix
        self.theta = np.zeros(N)   # Threshold vector

    def store_single_pattern(self, P):
        """
        Store a single pattern P using the given weight matrix formula.
        """
        P = np.array(P).flatten()  # Flatten the 2D pattern into a 1D array
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.W[i, j] = ((2 * P[i] - 1) * (2 * P[j] - 1)) / self.N
                else:
                    self.W[i, j] = 0  # Diagonal elements are zero

    def store_multiple_patterns(self, patterns):
        """
        Store multiple patterns using the Hebbian learning rule.
        """
        M = len(patterns)  # Number of patterns
        for P in patterns:
            P = np.array(P).flatten()  # Flatten the 2D pattern into a 1D array
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        self.W[i, j] += (2 * P[i] - 1) * (2 * P[j] - 1)
        self.W /= self.N  # Normalize by N

    def recover_pattern(self, X, max_iter=10):
        """
        Recover a pattern from a noisy input X using the weight matrix.
        """
        X = np.array(X).flatten()  # Flatten the input pattern
        S = X.copy()
        for _ in range(max_iter):
            U = np.dot(self.W, S) - self.theta
            S_new = np.sign(U)
            if np.array_equal(S_new, S):  # Convergence check
                break
            S = S_new
        return S

    def add_noise(self, pattern, noise_level):
        """
        Add noise to a pattern by flipping a fraction of its bits.
        """
        noisy_pattern = pattern.copy().flatten()  # Flatten the pattern
        num_flips = int(noise_level * self.N)
        flip_indices = np.random.choice(self.N, num_flips, replace=False)
        noisy_pattern[flip_indices] = 1 - noisy_pattern[flip_indices]  # Flip between 0 and 1
        return noisy_pattern


def preprocess_image(image_path):
    """
    Load an image, resize to 8x8, and convert to grayscale.
    """
    # Ensure the path is a string and exists
    if not isinstance(image_path, str):
        raise ValueError("File path must be a string.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    try:
        # Load the image using skimage
        image = io.imread(image_path)

        # If the image has an alpha channel (RGBA), remove it
        if image.shape[-1] == 4:  # Check if the image has 4 channels (RGBA)
            image = color.rgba2rgb(image)  # Convert RGBA to RGB

        # Convert the image to grayscale
        image_gray = color.rgb2gray(image)

        # Resize the image to the target size
        image_resized = transform.resize(image_gray, IMAGE_SIZE, anti_aliasing=0)

        return image_resized
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def convert_to_binary(image):
    """
    Convert a grayscale image to a binary matrix based on a threshold.
    """
    threshold = np.mean(image)  # Compute the threshold
    binary_matrix = (image > threshold).astype(int)  # Convert to binary
    binary_matrix[binary_matrix == 0] = -1  # Convert 0s to -1s for associative memory
    return binary_matrix

def display_binary_grid(binary_matrix):
    """
    Display the binary matrix in an 8x8 grid, using white and black squares.
    """
    for row in binary_matrix:
        # Map 1 to a white square (□) and -1 to a black square (■)
        row_str = " ".join("□" if x == 1 else "■" for x in row)
        print(row_str)

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

def store_pattern(image_path, name, am):
    """
    Preprocess an image, convert it to binary, and store it in the pattern dictionary with a custom name.
    If a pattern with the same name already exists, it will be replaced.
    Save the pattern to a text file and update the weight matrix.
    """
    print(f"Debug: Input file path: {image_path}")  # Debug statement

    try:
        # Preprocess the image and convert it to a binary matrix
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)

        # Store or replace the pattern in the dictionary
        PATTERN_STORE[name] = binary_matrix
        print(f"Pattern '{name}' stored successfully.")

        # Save the pattern to a text file
        if not os.path.exists(PATTERNS_DIR):
            os.makedirs(PATTERNS_DIR)  # Create the directory if it doesn't exist
        file_path = os.path.join(PATTERNS_DIR, f"{name}.txt")
        with open(file_path, "w") as file:
            for row in binary_matrix:
                file.write(" ".join(map(str, row)) + "\n")
        print(f"Pattern '{name}' saved to '{file_path}'.")

        # Update the weight matrix with all patterns
        patterns = list(PATTERN_STORE.values())
        am.store_multiple_patterns(patterns)
        print("Weight matrix updated with the new pattern.")
    except Exception as e:
        print(f"Error storing pattern: {e}")

        
def load_all_patterns_from_files(am):
    """
    Load all patterns from text files in the patterns directory and compute the weight matrix.
    """
    if not os.path.exists(PATTERNS_DIR):
        print(f"Error: Directory '{PATTERNS_DIR}' does not exist.")
        return

    # Clear the existing patterns
    PATTERN_STORE.clear()

    # Load patterns from text files
    for file_name in os.listdir(PATTERNS_DIR):
        if file_name.endswith(".txt"):
            file_path = os.path.join(PATTERNS_DIR, file_name)
            name = os.path.splitext(file_name)[0]  # Extract the pattern name from the file name
            try:
                # Read the binary matrix from the text file
                with open(file_path, "r") as file:
                    lines = file.readlines()
                    binary_matrix = []
                    for line in lines:
                        row = list(map(int, line.strip().split()))
                        binary_matrix.append(row)
                    binary_matrix = np.array(binary_matrix)

                # Store the pattern in the dictionary
                PATTERN_STORE[name] = binary_matrix
                print(f"Pattern '{name}' loaded from '{file_path}'.")
            except Exception as e:
                print(f"Error loading pattern from file '{file_path}': {e}")

    # Compute the weight matrix for the associative memory
    if PATTERN_STORE:
        patterns = list(PATTERN_STORE.values())
        am.store_multiple_patterns(patterns)
        print("Weight matrix computed from loaded patterns.")

def recognize_pattern(image_path, am):
    """
    Recognize a pattern by comparing the input image with stored patterns.
    Output the best-matched pattern.
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

            # Output the recognized pattern
            recognized_pattern = PATTERN_STORE[best_match]
            print("\nRecognized Pattern:")
            display_binary_grid(recognized_pattern)
        else:
            print("No patterns stored for recognition.")
    except Exception as e:
        print(f"Error recognizing pattern: {e}")

def recover_noisy_pattern(image_path, am, noise_level=0.5):
    """
    Recover a noisy pattern using the associative memory.
    """
    try:
        # Preprocess the input image
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)

        # Add noise to the pattern
        noisy_pattern = am.add_noise(binary_matrix.flatten(), noise_level)
        noisy_pattern = noisy_pattern.reshape(IMAGE_SIZE)

        print("Original Pattern:")
        display_binary_grid(binary_matrix)
        print("\nNoisy Pattern:")
        display_binary_grid(noisy_pattern)

        # Recover the pattern using associative memory
        recovered_pattern = am.recover_pattern(noisy_pattern.flatten())
        recovered_pattern = recovered_pattern.reshape(IMAGE_SIZE)

        print("\nRecovered Pattern:")
        display_binary_grid(recovered_pattern)

        # Compare recovered pattern with original
        hamming_distance = compare_images(binary_matrix, recovered_pattern)
        print(f"\nHamming Distance between Original and Recovered: {hamming_distance:.2%}")
    except Exception as e:
        print(f"Error recovering noisy pattern: {e}")

def main():
    am = AssociativeMemory(IMAGE_SIZE[0] * IMAGE_SIZE[1])  # Initialize associative memory

    # Load all patterns from text files at the start
    load_all_patterns_from_files(am)

    while True:
        print("\nMenu:")
        print("1. Store a pattern")
        print("2. Recognize a pattern")
        print("3. Recover a noisy pattern")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            image_path = input("Enter the path to the image file: ").strip()
            if not os.path.exists(image_path):
                print(f"Error: File '{image_path}' does not exist.")
                continue
            name = input("Enter a custom name for this pattern: ").strip()
            store_pattern(image_path, name)
            # Update the weight matrix after storing a new pattern
            patterns = list(PATTERN_STORE.values())
            am.store_multiple_patterns(patterns)
        elif choice == "2":
            image_path = input("Enter the path to the image file for recognition: ").strip()
            if not os.path.exists(image_path):
                print(f"Error: File '{image_path}' does not exist.")
                continue
            recognize_pattern(image_path, am)
        elif choice == "3":
            image_path = input("Enter the path to the image file for recovery: ").strip()
            if not os.path.exists(image_path):
                print(f"Error: File '{image_path}' does not exist.")
                continue
            recover_noisy_pattern(image_path, am, noise_level=0.2)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()