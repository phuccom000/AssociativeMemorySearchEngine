import numpy as np
import os
import warnings
import sys
sys.set_int_max_str_digits(10000)

warnings.filterwarnings("ignore", category=UserWarning)

PATTERN_STORE = {}
PATTERNS_DIR = "patterns"
pattern_shape = None  # Inferred dynamically from the first loaded pattern
PATTERN_CACHE_FILE = "pattern_cache.npz"


class AssociativeMemory:
    def __init__(self, size):
        self.size = size
        self.memory = np.zeros((size, size))

    def store(self, input_vector, output_vector):
        self.memory += np.outer(output_vector, input_vector)

    def store_multiple_patterns(self, patterns):
        for pattern in patterns:
            self.store(pattern, pattern)

    def recall(self, input_vector, threshold=0):
        recalled = np.dot(self.memory, input_vector)
        return np.where(recalled > threshold, 1, -1)


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    def store_multiple_patterns(self, patterns):
        self.W = np.zeros((self.size, self.size))
        for P in patterns:
            P = np.array(P).flatten()
            for i in range(self.size):
                for j in range(self.size):
                    if i != j:
                        self.W[i, j] += (2 * P[i] - 1) * (2 * P[j] - 1)
        self.W /= self.size
        np.fill_diagonal(self.W, 0)

    def recall(self, pattern, max_iter=100):
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


def add_noise(pattern, noise_level=0.1):
    random_mask = np.random.rand(*pattern.shape)
    return np.where(random_mask < noise_level, 1 - pattern, pattern)


def load_all_patterns_from_files(am, hn):
    global PATTERN_STORE, pattern_shape

    # Try to load from cache
    if os.path.exists(PATTERN_CACHE_FILE):
        try:
            print("Loading patterns from cache...")
            cache = np.load(PATTERN_CACHE_FILE, allow_pickle=True)
            PATTERN_STORE = cache['patterns'].item()
            pattern_shape = tuple(int(dim) for dim in binary_matrix.shape)
            print(f"Loaded {len(PATTERN_STORE)} patterns from cache with shape {pattern_shape}")

            total_size = int(np.prod(pattern_shape))
            am.__init__(total_size)
            hn.__init__(total_size)

            patterns_am = [(p * 2 - 1).flatten() for p in PATTERN_STORE.values()]
            patterns_hn = [p.flatten() for p in PATTERN_STORE.values()]
            am.store_multiple_patterns(patterns_am)
            hn.store_multiple_patterns(patterns_hn)
            return
        except Exception as e:
            print(f"Cache load failed: {e}. Reloading from files...")

    # Fall back to loading from pattern files
    PATTERN_STORE.clear()
    pattern_shape = None

    print("Loading patterns recursively from subfolders:")
    for root, _, files in os.walk(PATTERNS_DIR):
        for file_name in sorted(files):
            if not (file_name.endswith(".txt") or file_name.endswith(".npy")):
                continue

            path = os.path.join(root, file_name)
            try:
                if file_name.endswith(".npy"):
                    binary_matrix = np.load(path)
                else:  # .txt
                    with open(path, 'r') as f:
                        content = f.read().strip()
                        if all(c in '01' for c in content) and '\n' not in content:
                            binary_array = np.array([int(c) for c in content])
                            if pattern_shape is None:
                                pattern_shape = (len(binary_array), 1)
                            elif len(binary_array) != np.prod(pattern_shape):
                                print(f"Skipping {path}: size mismatch")
                                continue
                            binary_matrix = binary_array.reshape(pattern_shape)
                        else:
                            binary_matrix = np.loadtxt(path).astype(int)

                # Initialize pattern shape
                if pattern_shape is None:
                    pattern_shape = binary_matrix.shape
                elif binary_matrix.shape != pattern_shape:
                    print(f"Skipping {path}: shape mismatch")
                    continue

                # Determine display name
                relative_path = os.path.relpath(path, PATTERNS_DIR)
                display_name = os.path.splitext(relative_path)[0].replace(os.sep, '/')


                PATTERN_STORE[display_name] = binary_matrix
                print(f"  Loaded '{path}' as '{display_name}'")

            except Exception as e:
                print(f"  Error loading {path}: {e}")

    if PATTERN_STORE:
        total_size = int(np.prod(pattern_shape))
        am.__init__(total_size)
        hn.__init__(total_size)

        patterns_am = [(p * 2 - 1).flatten() for p in PATTERN_STORE.values()]
        patterns_hn = [p.flatten() for p in PATTERN_STORE.values()]
        am.store_multiple_patterns(patterns_am)
        hn.store_multiple_patterns(patterns_hn)

        # Save cache
        try:
            np.savez(PATTERN_CACHE_FILE, patterns=PATTERN_STORE, shape=pattern_shape)
            print(f"\nSaved pattern cache to '{PATTERN_CACHE_FILE}'")
        except Exception as e:
            print(f"Error saving cache: {e}")

        print(f"\nLoaded {len(PATTERN_STORE)} patterns from disk with shape {pattern_shape}")
    else:
        print("No valid patterns found.")



def recall_from_file(path, am, hn, noise_level=0.1):
    global pattern_shape

    try:
        if pattern_shape is None:
            print("Error: No reference pattern shape loaded.")
            return

        if path.endswith(".npy"):
            binary_matrix = np.load(path, mmap_mode='r')
        else:
            with open(path, 'r') as f:
                content = f.read().strip()
                if all(c in '01' for c in content) and '\n' not in content:
                    binary_array = np.array([int(c) for c in content])
                else:
                    binary_array = np.loadtxt(path).astype(int).flatten()

            if binary_array.ndim > 1:
                binary_matrix = binary_array.reshape(pattern_shape)
            elif binary_array.size == np.prod(pattern_shape):
                binary_matrix = binary_array.reshape(pattern_shape)
            else:
                print(f"Error: Input shape {binary_array.shape} does not match expected {pattern_shape}")
                return

        original = binary_matrix.flatten()
        noisy = add_noise(binary_matrix, noise_level).flatten()

        # Associative Memory recall
        am_result = am.recall((noisy * 2 - 1))
        am_bin = np.where(am_result == 1, 1, 0)

        # Hopfield Network recall
        hn_result = hn.recall((noisy * 2 - 1))
        hn_bin = np.where(hn_result == 1, 1, 0)

        # Calculate accuracy
        acc_am = np.mean(am_bin == original)
        acc_hn = np.mean(hn_bin == original)

        # Match to closest stored pattern (based on output)
        def find_best_match(recalled_bin):
            best_name = None
            best_score = -1
            for name, stored in PATTERN_STORE.items():
                flat_stored = stored.flatten()
                if flat_stored.shape != recalled_bin.shape:
                    continue
                score = np.mean(flat_stored == recalled_bin)
                if score > best_score:
                    best_score = score
                    best_name = name
            folder_name = best_name.split('/')[0] if best_name else "Unknown"
            return folder_name, best_score

        match_am, score_am = find_best_match(am_bin)
        match_hn, score_hn = find_best_match(hn_bin)

        print(f"\nRecall results for: {os.path.basename(path)}")
        print(f"  Accuracy (AM): {acc_am:.2f} | Closest match: {match_am} ({score_am:.2f})")
        print(f"  Accuracy (HN): {acc_hn:.2f} | Closest match: {match_hn} ({score_hn:.2f})")

    except Exception as e:
        print(f"Error during recall: {e}")

if __name__ == "__main__":
    print("Binary Associative Memory & Hopfield Network System\n")
    am = AssociativeMemory(0)
    hn = HopfieldNetwork(0)

    load_all_patterns_from_files(am, hn)

    while True:
        print("\nMenu:")
        print("1. Recall from binary file path")
        print("2. Clear pattern cache and reload")
        print("3. Exit")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            path = input("Enter path to binary .txt file: ").strip()
            if os.path.exists(path):
                level = float(input("Enter noise level (0.0 - 1.0): ").strip())
                recall_from_file(path, am, hn, level)
            else:
                print("File not found.")
        elif choice == "2":
            if os.path.exists(PATTERN_CACHE_FILE):
                os.remove(PATTERN_CACHE_FILE)
                print("Cache cleared.")
            load_all_patterns_from_files(am, hn)
        
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")
