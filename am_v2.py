import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import binarize
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
IMAGE_SIZE = (32, 32)  # 32x32 patterns
PATTERN_LENGTH = IMAGE_SIZE[0] * IMAGE_SIZE[1]
INPUT_IMAGES_DIR = "input_images"
PATTERNS_DIR = "patterns"  # Directory to store pattern files
PATTERN_STORE = {}  # Global dictionary to store patterns

class AssociativeMemory:
    def __init__(self, size):
        """Initialize the associative memory matrix."""
        self.size = size
        self.memory = np.zeros((size, size))

    def store(self, input_vector, output_vector):
        """Store an association using outer product (auto-associative)."""
        self.memory += np.outer(output_vector, input_vector)

    def store_multiple_patterns(self, patterns):
        """Store multiple patterns."""
        flattened_patterns = [p.flatten() for p in patterns]
        for pattern in flattened_patterns:
            self.store(pattern, pattern)

    def recall(self, input_vector, threshold=0):
        """Recall an output based on the input vector."""
        recalled = np.dot(self.memory, input_vector)
        return np.where(recalled > threshold, 1, -1)

class HopfieldNetwork:
    def __init__(self, size):
        """Initialize Hopfield network with given size."""
        self.size = size
        self.W = np.zeros((size, size))  # Weight matrix

    def store_single_pattern(self, P):
        """
        Store a single pattern P using the given weight matrix formula.
        """
        P = np.array(P).flatten()  # Flatten the 2D pattern into a 1D array
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.W[i, j] = ((2 * P[i] - 1) * (2 * P[j] - 1)) / self.size
                else:
                    self.W[i, j] = 0  # Diagonal elements are zero

    def store_multiple_patterns(self, patterns):
        """
        Store multiple patterns using the Hebbian learning rule (corrected for diagonal).
        """
        M = len(patterns)  # Number of patterns
        self.W = np.zeros((self.size, self.size))  # Initialize weights to zero

        for P in patterns:
            P = np.array(P).flatten()  # Flatten the 2D pattern into a 1D array
            for i in range(self.size):
                for j in range(self.size):
                    if i != j:
                        self.W[i, j] += (2 * P[i] - 1) * (2 * P[j] - 1)

        self.W /= self.size  # Normalize by size

        # Explicitly set the diagonal to zero
        for i in range(self.size):
            self.W[i, i] = 0

    def recall(self, pattern, max_iter=100):
        """Recall pattern using asynchronous updates."""
        pattern = pattern.copy()
        for _ in range(max_iter):
            updated = False
            for i in range(self.size):
                new_state = np.sign(np.dot(self.W[i], pattern))
                if new_state != pattern[i]:
                    pattern[i] = new_state
                    updated = True
            if not updated:
                break
        return pattern

    def energy(self, pattern):
        """Compute energy of the current state."""
        return -0.5 * np.dot(pattern, np.dot(self.W, pattern))

def preprocess_image(image_path, size=IMAGE_SIZE):
    """Load, resize, binarize and flatten an image."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(size, Image.NEAREST)
    img_array = np.array(img)
    binary_img = binarize(img_array.reshape(1, -1), threshold=127).reshape(size) # Reshape to 2D
    return binary_img * 2 - 1  # Convert to -1, 1 values

def convert_to_binary(image):
    """Convert a preprocessed image array (-1, 1) to binary (0, 1)."""
    return np.where(image == 1, 1, 0).astype(int)

def display_pattern(pattern, shape=IMAGE_SIZE):
    """Display a pattern as an image."""
    plt.imshow(pattern.reshape(shape), cmap='binary')
    plt.axis('off')
    plt.show()

def display_binary_grid(binary_matrix):
    """Display condensed 32x32 grid (showing every 4th pixel)"""
    if binary_matrix.shape != (32, 32):
        print("Error: Input matrix must be 32x32 for this display.")
        return
    for i in range(0, 32, 4):
        row = binary_matrix[i]
        print(" ".join("□" if x == 1 else "0" if x == 0 else "■" for x in row[::4]))

def add_noise(pattern, noise_level=0.1):
    """Add random noise to a pattern."""
    random_mask = np.random.rand(*pattern.shape)
    noisy_pattern = np.where(random_mask < noise_level, -pattern, pattern)
    return noisy_pattern

def store_pattern(image_path, name, am, hn):
    """Store a pattern from an image in both memories and save to file."""
    global PATTERN_STORE
    try:
        image = preprocess_image(image_path)
        binary_matrix = convert_to_binary(image)

        PATTERN_STORE[name] = binary_matrix
        print(f"Pattern '{name}' stored successfully.")

        if not os.path.exists(PATTERNS_DIR):
            os.makedirs(PATTERNS_DIR)
        np.savetxt(os.path.join(PATTERNS_DIR, f"{name}.txt"), binary_matrix, fmt='%d')

        patterns = list(PATTERN_STORE.values())
        am.store_multiple_patterns([(p * 2 - 1).flatten() for p in patterns])
        hn.store_multiple_patterns(patterns) # Store in Hopfield using multiple patterns
        print(f"Total patterns stored: {len(patterns)}")
    except Exception as e:
        print(f"Error storing pattern: {e}")

def load_all_patterns_from_files(am, hn):
    """Load patterns from the patterns directory into both memories."""
    if not os.path.exists(PATTERNS_DIR):
        print(f"Directory '{PATTERNS_DIR}' does not exist.")
        return

    global PATTERN_STORE
    PATTERN_STORE.clear()

    for file_name in os.listdir(PATTERNS_DIR):
        if file_name.endswith(".txt"):
            try:
                binary_matrix = np.loadtxt(os.path.join(PATTERNS_DIR, file_name)).astype(int)
                name = os.path.splitext(file_name)[0]
                PATTERN_STORE[name] = binary_matrix
                print(f"Loaded pattern '{name}' from file.")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    if PATTERN_STORE:
        patterns_for_am = [(p * 2 - 1).flatten() for p in PATTERN_STORE.values()]
        am.store_multiple_patterns(patterns_for_am)
        hn.store_multiple_patterns(list(PATTERN_STORE.values()))

def recognize_pattern(image_path, am):
    """Recognize a pattern from an image using Associative Memory."""
    global PATTERN_STORE
    try:
        image = preprocess_image(image_path)
        binary_input = convert_to_binary(image)
        input_pattern = (binary_input * 2 - 1).flatten() # Convert to -1, 1 for recall
        recalled_pattern = am.recall(input_pattern)

        print("\nInput Pattern (Condensed for AM):")
        display_binary_grid(binary_input)

        recalled_binary = np.where(recalled_pattern == 1, 1, 0).reshape(IMAGE_SIZE)
        print("\nRecalled Pattern (Condensed for AM):")
        display_binary_grid(recalled_binary)

        best_match = None
        max_similarity = -1
        for name, stored_binary in PATTERN_STORE.items():
            similarity = np.mean(recalled_binary == stored_binary)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name

        if best_match:
            print(f"\nRecognized by AM as: '{best_match}' with similarity {max_similarity:.2f}")
        else:
            print("\nNo close match found in stored patterns (AM).")

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'.")
    except Exception as e:
        print(f"Error recognizing pattern: {e}")

def recover_noisy_pattern_am(image_path, am, noise_level):
    """Recover a noisy pattern from an image using Associative Memory and display."""
    try:
        original_image = preprocess_image(image_path)
        original_binary = convert_to_binary(original_image)
        original_pattern = (original_binary * 2 - 1).flatten()
        noisy_pattern = add_noise(original_pattern, noise_level)
        noisy_binary = np.where(noisy_pattern == 1, 1, 0).reshape(IMAGE_SIZE)
        recalled_am = am.recall(noisy_pattern)
        recalled_am_binary = np.where(recalled_am == 1, 1, 0).reshape(IMAGE_SIZE)
        accuracy_am = np.mean(recalled_am_binary == original_binary)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(original_binary, cmap='binary')
        plt.title('Original Pattern')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(noisy_binary, cmap='binary')
        plt.title(f'Noisy Pattern ({noise_level*100:.0f}% noise)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(recalled_am_binary, cmap='binary')
        plt.title(f'Recovered (AM)\nAccuracy: {accuracy_am:.2f}')
        plt.axis('off')

        plt.tight_layout()

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"noise{noise_level:.2f}_am_{base_name}.png"
        plt.savefig(filename)
        plt.show()
        plt.close()

        return accuracy_am

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'.")
        return None
    except Exception as e:
        print(f"Error recovering noisy pattern (AM): {e}")
        return None

def recover_noisy_pattern_hopfield(image_path, hn, noise_level):
    """Recover a noisy pattern from an image using Hopfield Network and display."""
    try:
        original_image = preprocess_image(image_path)
        original_binary = convert_to_binary(original_image)
        original_pattern = (original_binary * 2 - 1).flatten()
        noisy_pattern = add_noise(original_pattern, noise_level)
        noisy_binary = np.where(noisy_pattern == 1, 1, 0).reshape(IMAGE_SIZE)
        recalled_hn = hn.recall(noisy_pattern)
        recalled_hn_binary = np.where(recalled_hn == 1, 1, 0).reshape(IMAGE_SIZE)
        accuracy_hn = np.mean(recalled_hn_binary == original_binary)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(original_binary, cmap='binary')
        plt.title('Original Pattern')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(noisy_binary, cmap='binary')
        plt.title(f'Noisy Pattern ({noise_level*100:.0f}% noise)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(recalled_hn_binary, cmap='binary')
        plt.title(f'Recovered (HN)\nAccuracy: {accuracy_hn:.2f}')
        plt.axis('off')

        plt.tight_layout()

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"noise{noise_level:.2f}_hn_{base_name}.png"
        plt.savefig(filename)
        plt.show()

        plt.close()

        return accuracy_hn

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'.")
        return None
    except Exception as e:
        print(f"Error recovering noisy pattern (Hopfield): {e}")
        return None

def recover_all_noisy_patterns(noise_level, am, hn, restore_iterations=1):
    """Recovers all stored patterns with added noise, restores multiple times,
    calculates average accuracy, and then compares results."""
    if not PATTERN_STORE:
        print("No patterns stored to recover.")
        return

    am_avg_accuracies = {}
    hn_avg_accuracies = {}

    print(f"\nRecovering all patterns with noise level: {noise_level:.2f}")
    print(f"Restoring each noisy pattern {restore_iterations} times and averaging accuracy.")

    for name, original_binary in PATTERN_STORE.items():
        original_pattern = (original_binary * 2 - 1).flatten()
        am_accuracies_per_pattern = []
        hn_accuracies_per_pattern = []

        for _ in range(restore_iterations):
            noisy_pattern = add_noise(original_pattern, noise_level)

            # Recover with Associative Memory
            recalled_am = am.recall(noisy_pattern)
            recalled_am_binary = np.where(recalled_am == 1, 1, 0).reshape(IMAGE_SIZE)
            accuracy_am = np.mean(recalled_am_binary == original_binary)
            am_accuracies_per_pattern.append(accuracy_am)

            # Recover with Hopfield Network
            noisy_pattern_hn = noisy_pattern.copy() # Important to use a copy for HN recall
            recalled_hn_flattened = hn.recall(noisy_pattern_hn)
            recalled_hn_binary = np.where(recalled_hn_flattened == 1, 1, 0).reshape(IMAGE_SIZE)
            accuracy_hn = np.mean(recalled_hn_binary == original_binary)
            hn_accuracies_per_pattern.append(accuracy_hn)

        am_avg_accuracies[name] = np.mean(am_accuracies_per_pattern)
        hn_avg_accuracies[name] = np.mean(hn_accuracies_per_pattern)
        print(f"  Average AM Recovery Accuracy for '{name}': {am_avg_accuracies[name]:.2f}")
        print(f"  Average HN Recovery Accuracy for '{name}': {hn_avg_accuracies[name]:.2f}")

    # Plotting the results
    pattern_names = list(PATTERN_STORE.keys())
    am_avg_acc_values = list(am_avg_accuracies.values())
    hn_avg_acc_values = list(hn_avg_accuracies.values())

    x = np.arange(len(pattern_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, am_avg_acc_values, width, label='Associative Memory')
    rects2 = ax.bar(x + width/2, hn_avg_acc_values, width, label='Hopfield Network')

    ax.set_ylabel('Average Recovery Accuracy')
    ax.set_xlabel('Pattern')
    ax.set_title(f'Average Recovery Accuracy at Noise Level {noise_level:.2f} (averaged over {restore_iterations} restorations)')
    ax.set_xticks(x)
    ax.set_xticklabels(pattern_names, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    filename = f"noise{noise_level:.2f}_restore{restore_iterations}_all.png"
    plt.savefig(filename)
    plt.show()

    plt.close()

def convert_images_to_patterns():
    """Converts images in the input_images folder to patterns and saves them."""
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"Directory '{INPUT_IMAGES_DIR}' does not exist.")
        return

    if not os.path.exists(PATTERNS_DIR):
        os.makedirs(PATTERNS_DIR)
        print(f"Created directory '{PATTERNS_DIR}' for storing patterns.")

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No image files found in '{INPUT_IMAGES_DIR}'.")
        return

    print("Converting images to patterns:")
    for image_file in image_files:
        image_path = os.path.join(INPUT_IMAGES_DIR, image_file)
        name = os.path.splitext(image_file)[0]
        pattern_file = os.path.join(PATTERNS_DIR, f"{name}.txt")
        if not os.path.exists(pattern_file):
            try:
                image = preprocess_image(image_path)
                binary_matrix = convert_to_binary(image)
                PATTERN_STORE[name] = binary_matrix
                np.savetxt(pattern_file, binary_matrix, fmt='%d')
                print(f"  Converted '{image_file}' to pattern '{name}'.txt")
            except Exception as e:
                print(f"  Error converting '{image_file}': {e}")
        else:
            print(f"  Pattern '{name}.txt' already exists. Skipping conversion.")

if __name__ == "__main__":
    am = AssociativeMemory(IMAGE_SIZE[0] * IMAGE_SIZE[1])
    hn = HopfieldNetwork(IMAGE_SIZE[0] * IMAGE_SIZE[1])

    # Create patterns directory if it doesn't exist
    if not os.path.exists(PATTERNS_DIR):
        os.makedirs(PATTERNS_DIR)
        print(f"Created directory '{PATTERNS_DIR}' for storing patterns.")

    # Auto load images to patterns on startup (without overwriting existing)
    convert_images_to_patterns()

    # Load all patterns from files
    load_all_patterns_from_files(am, hn)

    # Limit to 15 patterns for stability
    MAX_PATTERNS = 15
    if len(PATTERN_STORE) > MAX_PATTERNS:
        print(f"Keeping first {MAX_PATTERNS} patterns for stability")
        limited_pattern_store = dict(list(PATTERN_STORE.items())[:MAX_PATTERNS])
        PATTERN_STORE = limited_pattern_store
        patterns_for_am = [(p * 2 - 1).flatten() for p in PATTERN_STORE.values()]
        am.store_multiple_patterns(patterns_for_am)
        hn.store_multiple_patterns(list(PATTERN_STORE.values()))

    while True:
        print("\nMenu:")
        print("1. Store a pattern from image")
        print("2. Recognize a pattern (Associative Memory)")
        print("3. Recover a noisy pattern (Associative Memory)")
        print("4. Recover a noisy pattern (Hopfield Network)")
        print("5. Recover all noisy patterns and compare")
        print("6. Exit")

        choice = input("Enter choice (1-6): ").strip()

        if choice == "1":
            image_path = input("Image path to store: ").strip()
            if os.path.exists(image_path):
                name = input("Pattern name: ").strip()
                store_pattern(image_path, name, am, hn)
            else:
                print("File not found")
        elif choice == "2":
            image_path = input("Image path to recognize: ").strip()
            if os.path.exists(image_path):
                recognize_pattern(image_path, am)
            else:
                print("File not found")
        elif choice == "3":
            image_path = input("Image path to recover (will add noise for AM): ").strip()
            if os.path.exists(image_path):
                noise_level_am = float(input("Enter noise level (0.0-1.0) for AM: ").strip())
                recover_noisy_pattern_am(image_path, am, noise_level_am)
            else:
                print("File not found")
        elif choice == "4":
            image_path = input("Image path to recover (will add noise for Hopfield): ").strip()
            if os.path.exists(image_path):
                noise_level_hn = float(input("Enter noise level (0.0-1.0) for Hopfield: ").strip())
                recover_noisy_pattern_hopfield(image_path, hn, noise_level_hn)
            else:
                print("File not found")
        elif choice == "5":
            noise_level_all = float(input("Enter noise level (0.0-1.0) for recovery comparison: ").strip())
            iteration = int(input("Enter how many recovery should be made to calculate average: ").strip())
            recover_all_noisy_patterns(noise_level_all, am, hn, iteration)
        elif choice == "6":
            print("Exiting program")
            break
        else:
            print("Invalid choice, please enter 1-6")