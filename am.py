import numpy as np
from skimage import io, color, transform
import os
import warnings
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
IMAGE_SIZE = (32, 32)  # 32x32 patterns for better digit recognition
PATTERN_STORE = {}      # Global dictionary for patterns
PATTERNS_DIR = "patterns"

class AssociativeMemory:
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N))
        self.theta = np.zeros(N)
        self.patterns = []

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

    def recover_pattern(self, X, max_iter=50, noise_tolerance=0.1):
        X = np.array(X).flatten()
        S = X.copy()
        
        for _ in range(max_iter):
            update_order = np.random.permutation(self.N)
            changed = False
            
            for i in update_order:
                U = np.dot(self.W[i,:], S) - self.theta[i]
                new_val = 1 if U > 0 else -1
                
                if abs(new_val - S[i]) > noise_tolerance:
                    S[i] = new_val
                    changed = True
            
            if not changed:
                break
                
        return S

    def add_noise(self, pattern, noise_level):
        noisy_pattern = pattern.copy().flatten()
        num_flips = min(int(noise_level * self.N), self.N)
        flip_indices = np.random.choice(self.N, num_flips, replace=False)
        noisy_pattern[flip_indices] = -noisy_pattern[flip_indices]
        return noisy_pattern

def preprocess_image(image_path):
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        raise ValueError("Invalid file path")
    
    try:
        image = io.imread(image_path)
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        image_gray = color.rgb2gray(image)
        return transform.resize(image_gray, IMAGE_SIZE, order=0, anti_aliasing=False, preserve_range=True)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def convert_to_binary(image):
    threshold = np.mean(image)  # Compute the threshold
    binary_matrix = (image > threshold).astype(int)  # Convert to binary
    binary_matrix[binary_matrix == 0] = -1  # Convert 0s to -1s for associative memory
    return binary_matrix

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

def display_binary_grid(binary_matrix):
    """Display condensed 32x32 grid (showing every 4th pixel)"""
    for i in range(0, 32, 4):
        row = binary_matrix[i]
        print(" ".join("□" if x == 1 else "■" for x in row[::4]))

def store_pattern(image_path, name, am):
    try:
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)
        
        PATTERN_STORE[name] = binary_matrix
        print(f"Pattern '{name}' stored successfully.")
        
        if not os.path.exists(PATTERNS_DIR):
            os.makedirs(PATTERNS_DIR)
        np.savetxt(os.path.join(PATTERNS_DIR, f"{name}.txt"), binary_matrix, fmt='%d')
        
        patterns = list(PATTERN_STORE.values())
        am.store_multiple_patterns(patterns)
        print(f"Total patterns stored: {len(patterns)}")
    except Exception as e:
        print(f"Error storing pattern: {e}")

def load_all_patterns_from_files(am):
    if not os.path.exists(PATTERNS_DIR):
        print(f"Directory '{PATTERNS_DIR}' does not exist.")
        return
    
    global PATTERN_STORE
    PATTERN_STORE.clear()
    
    for file_name in os.listdir(PATTERNS_DIR):
        if file_name.endswith(".txt"):
            try:
                binary_matrix = np.loadtxt(os.path.join(PATTERNS_DIR, file_name))
                name = os.path.splitext(file_name)[0]
                PATTERN_STORE[name] = binary_matrix
                print(f"Loaded pattern '{name}'")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    if PATTERN_STORE:
        am.store_multiple_patterns(list(PATTERN_STORE.values()))

def recognize_pattern(image_path, am):
    try:
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)
        
        best_match = None
        min_distance = float('inf')
        
        for name, pattern in PATTERN_STORE.items():
            distance = np.mean(binary_matrix != pattern)
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        if best_match:
            print(f"\nBest match: '{best_match}' (Difference: {min_distance:.1%})")
            print("Stored pattern:")
            display_binary_grid(PATTERN_STORE[best_match])
        else:
            print("No patterns available for recognition")
    except Exception as e:
        print(f"Recognition error: {e}")

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
    global PATTERN_STORE  # Declare global at function start
    
    am = AssociativeMemory(IMAGE_SIZE[0] * IMAGE_SIZE[1])
    load_all_patterns_from_files(am)
    
    # Limit to 15 patterns for stability
    MAX_PATTERNS = 15
    if len(PATTERN_STORE) > MAX_PATTERNS:
        print(f"Keeping first {MAX_PATTERNS} patterns for stability")
        PATTERN_STORE = dict(list(PATTERN_STORE.items())[:MAX_PATTERNS])
        am.store_multiple_patterns(list(PATTERN_STORE.values()))
    
    while True:
        print("\nMenu:")
        print("1. Store a pattern")
        print("2. Recognize a pattern")
        print("3. Recover a noisy pattern")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Image path: ").strip()
            if os.path.exists(image_path):
                name = input("Pattern name: ").strip()
                store_pattern(image_path, name, am)
            else:
                print("File not found")
        elif choice == "2":
            image_path = input("Image to recognize: ").strip()
            if os.path.exists(image_path):
                recognize_pattern(image_path, am)
        elif choice == "3":
            image_path = input("Image to recover: ").strip()
            if os.path.exists(image_path):
                recover_noisy_pattern(image_path, am, 0.1)
        elif choice == "4":
            print("Exiting program")
            break
        else:
            print("Invalid choice, please enter 1-4")

if __name__ == "__main__":
    main()