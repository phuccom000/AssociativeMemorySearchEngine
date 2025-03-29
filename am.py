import numpy as np
from skimage import io, color, transform
import os
import warnings
from scipy.linalg import orth
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
IMAGE_SIZE = (32, 32)  # 32x32 patterns for better digit recognition
PATTERN_STORE = {}      # Global dictionary for patterns
PATTERNS_DIR = "patterns"

class EnhancedAssociativeMemory:
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N))
        self.theta = np.zeros(N)
        self.patterns = []
        self.orthogonalized = False
    
    def store_single_pattern(self, P, orthogonalize=False):
        """
        Store a single pattern with optional orthogonalization
        """
        P = np.array(P).flatten()
        if orthogonalize and self.patterns:
            # Orthogonalize against existing patterns
            pattern_matrix = np.array(self.patterns).T
            new_pattern = P - np.dot(pattern_matrix, np.dot(pattern_matrix.T, P))
            P = np.sign(new_pattern)  # Binarize the result
        
        self.patterns.append(P.copy())
        
        # Update weight matrix using Hebbian learning
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.W[i, j] += (2 * P[i] - 1) * (2 * P[j] - 1)
        
        # Normalize weights by number of patterns
        if len(self.patterns) > 0:
            self.W = self.W / len(self.patterns)
    
    def store_multiple_patterns(self, patterns, orthogonalize=False):
        """
        Store multiple patterns with optional orthogonalization
        """
        self.orthogonalized = orthogonalize
        
        # Convert patterns to bipolar (-1, 1) and flatten
        patterns = [p.flatten() for p in patterns]
        patterns = [2*(p > 0) - 1 for p in patterns]  # Convert to -1,1
        
        if orthogonalize:
            # Convert patterns to orthogonal set using Gram-Schmidt
            pattern_matrix = np.array(patterns).T
            orth_patterns = orth(pattern_matrix)
            
            # Binarize the orthogonal patterns
            orth_patterns = np.sign(orth_patterns)
            patterns = [orth_patterns[:,i] for i in range(orth_patterns.shape[1])]
        
        self.patterns = patterns.copy()
        
        # Update weight matrix using Hebbian learning
        self.W = np.zeros((self.N, self.N))
        for P in patterns:
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        self.W[i, j] += (2 * P[i] - 1) * (2 * P[j] - 1)
        
        # Normalize weights by number of patterns
        if len(patterns) > 0:
            self.W = self.W / len(patterns)
    
    def recover_pattern(self, X, max_iter=50, noise_tolerance=0.1, async_update=True):
        """
        Enhanced pattern recovery with orthogonalization support
        """
        X = np.array(X).flatten()
        S = X.copy()
        
        for _ in range(max_iter):
            old_S = S.copy()
            
            if async_update:
                # Asynchronous update (better convergence)
                update_order = np.random.permutation(self.N)
                for i in update_order:
                    U = np.dot(self.W[i,:], S) - self.theta[i]
                    S[i] = 1 if U > 0 else -1
            else:
                # Synchronous update (faster but may oscillate)
                U = np.dot(self.W, S) - self.theta
                S = np.where(U > 0, 1, -1)
            
            # Stop if pattern has converged
            if np.array_equal(S, old_S):
                break
                
        return S
    
    def recognize_number(self, X, threshold=0.85):
        """
        Recognize which stored number the pattern matches
        """
        recalled = self.recover_pattern(X)
        
        # Calculate similarity with all stored patterns
        similarities = [np.dot(recalled, p) / self.N for p in self.patterns]
        max_sim = max(similarities)
        best_match = np.argmax(similarities)
        
        if max_sim >= threshold:
            return best_match
        return -1  # No match found
    
    def add_noise(self, pattern, noise_level):
        noisy_pattern = pattern.copy().flatten()
        num_flips = min(int(noise_level * self.N), self.N)
        flip_indices = np.random.choice(self.N, num_flips, replace=False)
        noisy_pattern[flip_indices] = -noisy_pattern[flip_indices]
        return noisy_pattern
    
    def energy(self, pattern):
        """Calculate the energy of a pattern"""
        return -0.5 * np.dot(pattern, np.dot(self.W, pattern))

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
    """
    if binary_matrix1.shape != binary_matrix2.shape:
        raise ValueError("Binary matrices must have the same dimensions for comparison.")

    flat1 = binary_matrix1.flatten()
    flat2 = binary_matrix2.flatten()

    hamming_distance = np.sum(flat1 != flat2) / len(flat1)
    similarity = np.dot(flat1, flat2) / len(flat1)
    return hamming_distance, similarity

def display_binary_grid(binary_matrix):
    """Display condensed 32x32 grid (showing every 4th pixel)"""
    for i in range(0, 32, 4):
        row = binary_matrix[i]
        print(" ".join("□" if x == 1 else "■" for x in row[::4]))

def store_pattern(image_path, name, am, orthogonalize=False):
    try:
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)
        
        PATTERN_STORE[name] = binary_matrix
        print(f"Pattern '{name}' stored successfully.")
        
        if not os.path.exists(PATTERNS_DIR):
            os.makedirs(PATTERNS_DIR)
        np.savetxt(os.path.join(PATTERNS_DIR, f"{name}.txt"), binary_matrix, fmt='%d')
        
        patterns = list(PATTERN_STORE.values())
        am.store_multiple_patterns(patterns, orthogonalize)
        print(f"Total patterns stored: {len(patterns)}")
    except Exception as e:
        print(f"Error storing pattern: {e}")

def load_all_patterns_from_files(am, orthogonalize=False):
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
        am.store_multiple_patterns(list(PATTERN_STORE.values()), orthogonalize)

def recognize_pattern(image_path, am):
    try:
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)
        
        # Use the enhanced recognition
        recognized_idx = am.recognize_number(binary_matrix)
        
        if recognized_idx >= 0:
            best_match = list(PATTERN_STORE.keys())[recognized_idx]
            hamming_distance, similarity = compare_images(binary_matrix, PATTERN_STORE[best_match])
            print(f"\nBest match: '{best_match}' (Similarity: {similarity:.1%})")
            print("Stored pattern:")
            display_binary_grid(PATTERN_STORE[best_match])
        else:
            print("No matching pattern found (below similarity threshold)")
    except Exception as e:
        print(f"Recognition error: {e}")

def recover_noisy_pattern(image_path, am, noise_level=0.3):
    """
    Enhanced noisy pattern recovery with orthogonalization support
    """
    try:
        # Preprocess the input image
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)

        # Add noise to the pattern
        noisy_pattern = am.add_noise(binary_matrix, noise_level)
        noisy_pattern = noisy_pattern.reshape(IMAGE_SIZE)

        print("\nOriginal Pattern:")
        display_binary_grid(binary_matrix)
        print("\nNoisy Pattern:")
        display_binary_grid(noisy_pattern)

        # Recover the pattern using associative memory
        recovered_pattern = am.recover_pattern(noisy_pattern.flatten())
        recovered_pattern = recovered_pattern.reshape(IMAGE_SIZE)

        print("\nRecovered Pattern:")
        display_binary_grid(recovered_pattern)

        # Compare recovered pattern with original
        hamming_distance, similarity = compare_images(binary_matrix, recovered_pattern)
        print(f"\nSimilarity between Original and Recovered: {similarity:.1%}")
        print(f"Energy of recovered pattern: {am.energy(recovered_pattern.flatten()):.2f}")
        
        # Try to recognize the recovered pattern
        recognized_idx = am.recognize_number(recovered_pattern)
        if recognized_idx >= 0:
            print(f"Recognized as: {list(PATTERN_STORE.keys())[recognized_idx]}")
    except Exception as e:
        print(f"Error recovering noisy pattern: {e}")

def main():
    global PATTERN_STORE
    
    am = EnhancedAssociativeMemory(IMAGE_SIZE[0] * IMAGE_SIZE[1])
    
    # Ask if user wants to use orthogonalization
    use_orth = input("Use pattern orthogonalization? (y/n): ").lower() == 'y'
    load_all_patterns_from_files(am, use_orth)
    
    # Limit to 12 patterns for stability (orthogonalization allows more)
    MAX_PATTERNS = 12 if use_orth else 8
    if len(PATTERN_STORE) > MAX_PATTERNS:
        print(f"Keeping first {MAX_PATTERNS} patterns for stability")
        PATTERN_STORE = dict(list(PATTERN_STORE.items())[:MAX_PATTERNS])
        am.store_multiple_patterns(list(PATTERN_STORE.values()), use_orth)
    
    while True:
        print("\nMenu:")
        print("1. Store a pattern")
        print("2. Recognize a pattern")
        print("3. Recover a noisy pattern")
        print("4. View system information")
        print("5. Exit")
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == "1":
            image_path = input("Image path: ").strip()
            if os.path.exists(image_path):
                name = input("Pattern name: ").strip()
                store_pattern(image_path, name, am, use_orth)
            else:
                print("File not found")
        elif choice == "2":
            image_path = input("Image to recognize: ").strip()
            if os.path.exists(image_path):
                recognize_pattern(image_path, am)
        elif choice == "3":
            image_path = input("Image to recover: ").strip()
            if os.path.exists(image_path):
                noise_level = float(input("Noise level (0-1): ").strip())
                recover_noisy_pattern(image_path, am, noise_level)
        elif choice == "4":
            print("\nSystem Information:")
            print(f"Pattern size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} ({IMAGE_SIZE[0]*IMAGE_SIZE[1]} neurons)")
            print(f"Stored patterns: {len(PATTERN_STORE)}/{MAX_PATTERNS}")
            print(f"Orthogonalization: {'Enabled' if am.orthogonalized else 'Disabled'}")
            if PATTERN_STORE:
                print("\nStored pattern energies:")
                for name, pattern in PATTERN_STORE.items():
                    print(f"{name}: {am.energy(pattern.flatten()):.2f}")
        elif choice == "5":
            print("Exiting program")
            break
        else:
            print("Invalid choice, please enter 1-5")

if __name__ == "__main__":
    main()