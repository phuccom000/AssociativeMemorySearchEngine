import numpy as np
from skimage import io, color, transform

# Constants
IMAGE_SIZE = (8, 8)  # Target size for the image (8x8)
PATTERN_STORE = {}   # Dictionary to store patterns with custom names

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
        P = np.array(P)
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
            P = np.array(P)
            for i in range(self.N):
                for j in range(self.N):
                    self.W[i, j] += (2 * P[i] - 1) * (2 * P[j] - 1)
        self.W /= self.N  # Normalize by N

    def recover_pattern(self, X, max_iter=10):
        """
        Recover a pattern from a noisy input X using the weight matrix.
        """
        X = np.array(X)
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
        noisy_pattern = pattern.copy()
        num_flips = int(noise_level * self.N)
        flip_indices = np.random.choice(self.N, num_flips, replace=False)
        noisy_pattern[flip_indices] *= -1  # Flip the selected bits
        return noisy_pattern


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
    binary_matrix[binary_matrix == 0] = -1  # Convert 0s to -1s for associative memory
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

def recognize_pattern(image_path, am):
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

def recover_noisy_pattern(image_path, am, noise_level=0.3):
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
    while True:
        print("\nMenu:")
        print("1. Store a pattern")
        print("2. Recognize a pattern")
        print("3. Recover a noisy pattern")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            image_path = input("Enter the path to the image file: ").strip()
            name = input("Enter a custom name for this pattern: ").strip()
            store_pattern(image_path, name)
        elif choice == "2":
            image_path = input("Enter the path to the image file for recognition: ").strip()
            recognize_pattern(image_path, am)
        elif choice == "3":
            image_path = input("Enter the path to the image file for recovery: ").strip()
            recover_noisy_pattern(image_path, am)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()