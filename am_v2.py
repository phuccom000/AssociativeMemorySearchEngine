import numpy as np
import os
import warnings
import sys
import matplotlib.pyplot as plt

sys.set_int_max_str_digits(10000)
warnings.filterwarnings("ignore", category=UserWarning)

PATTERN_STORE = {}
PATTERN_SHAPES = {}
PATTERNS_DIR = "patterns"
pattern_shape = None  # Inferred dynamically from the first loaded pattern
PATTERN_CACHE_FILE = "pattern_cache.npz"
DOWNSAMPLED_DIR = "patterns_downsampled"
NPY_CONVERTED_DIR = "patterns_npy"
DOWNSAMPLE_THRESHOLD = 1025  # Only downsample if pattern size exceeds this
DOWNSAMPLE_FACTOR = 10        # Reduce size by this factor

def downsample_vector(vector, factor=DOWNSAMPLE_FACTOR):
    length = len(vector)
    remainder = length % factor
    if remainder != 0:
        pad_size = factor - remainder
        padding = np.zeros(pad_size, dtype=np.int8)
        vector = np.concatenate([vector, padding])
        print(f"    ↳ Padded with {pad_size} zeros to length: {len(vector)}")

    reshaped = vector.reshape(-1, factor)
    pooled = (np.sum(reshaped, axis=1) >= (factor // 2)).astype(np.int8)
    return pooled


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
    global PATTERN_STORE, PATTERN_SHAPES, pattern_shape

    if os.path.exists(PATTERN_CACHE_FILE):
        try:
            print("Loading patterns from cache...")
            cache = np.load(PATTERN_CACHE_FILE, allow_pickle=True)
            PATTERN_STORE = cache['patterns'].item()
            pattern_shape = tuple(cache['shape'])
            PATTERN_SHAPES = cache['shapes'].item() if 'shapes' in cache else {}
            print(f"Loaded {len(PATTERN_STORE)} patterns from cache with shape {pattern_shape}")

            total_size = int(np.prod(pattern_shape))
            am.__init__(total_size)
            hn.__init__(total_size)

            if 'am_matrix' in cache:
                am.memory = cache['am_matrix']
                print("Loaded AM matrix from cache.")
            else:
                print("Warning: AM matrix not found in cache, recalculating.")
                patterns_am = [(p * 2 - 1).flatten() for p in PATTERN_STORE.values()]
                am.store_multiple_patterns(patterns_am)

            if 'hn_matrix' in cache:
                hn.W = cache['hn_matrix']
                print("Loaded HN matrix from cache.")
            else:
                print("Warning: HN matrix not found in cache, recalculating.")
                patterns_hn = [p.flatten() for p in PATTERN_STORE.values()]
                hn.store_multiple_patterns(patterns_hn)
            return
        except Exception as e:
            print(f"Cache load failed: {e}. Reloading from files...")

    # Start fresh if cache load failed
    PATTERN_STORE.clear()
    PATTERN_SHAPES.clear()
    pattern_shape = None

    print("Loading patterns recursively from subfolders:")
    for root, _, files in os.walk(PATTERNS_DIR):
        for file_name in sorted(files):
            if not (file_name.endswith(".txt") or file_name.endswith(".npy")):
                continue

            path = os.path.join(root, file_name)
            try:
                # Convert .txt to .npy if needed
                if file_name.endswith(".txt"):
                    with open(path, 'r') as f:
                        content = f.read().strip()
                        if all(c in '01' for c in content) and '\n' not in content:
                            binary_array = np.fromiter((int(c) for c in content), dtype=np.int8)
                        else:
                            binary_array = np.loadtxt(path).astype(int).flatten()
                    binary_array = binary_array.flatten()
                    print(f"  Loaded '{path}' | Original length: {len(binary_array)}")

                    # Save converted version to .npy
                    rel_path = os.path.relpath(path, PATTERNS_DIR)
                    npy_path = os.path.join(NPY_CONVERTED_DIR, os.path.splitext(rel_path)[0] + ".npy")
                    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                    np.save(npy_path, binary_array)
                    print(f"  Converted '{path}' → '{npy_path}'")
                    path = npy_path

                # Load .npy file
                binary_matrix = np.load(path).flatten()
                print(f"  Loaded '{path}' | Original length: {len(binary_matrix)}")

                if binary_matrix.size >= DOWNSAMPLE_THRESHOLD:
                    binary_matrix = downsample_vector(binary_matrix, factor=DOWNSAMPLE_FACTOR)
                    print(f"    ↳ Downsampled to length: {len(binary_matrix)}")

                    rel_path = os.path.relpath(path, PATTERNS_DIR)
                    down_path = os.path.join(DOWNSAMPLED_DIR, os.path.splitext(rel_path)[0] + ".npy")
                    os.makedirs(os.path.dirname(down_path), exist_ok=True)
                    np.save(down_path, binary_matrix)
                    print(f"    ↳ Saved downsampled to: {down_path}")

                if pattern_shape is None:
                    pattern_shape = (binary_matrix.size,)
                elif binary_matrix.shape != pattern_shape:
                    print(f"  Skipping {path}: shape mismatch")
                    continue

                # Use outermost folder name as base key
                # Determine if file is directly in PATTERNS_DIR
                rel_root = os.path.relpath(root, PATTERNS_DIR)
                if rel_root == ".":
                    base_key = os.path.splitext(file_name)[0]
                else:
                    outer_folder = rel_root.split(os.sep)[0]
                    base_key = outer_folder

                # Generate unique key if needed
                storage_key = base_key
                counter = 1
                while storage_key in PATTERN_STORE:
                    storage_key = f"{base_key}_{counter}"
                    counter += 1


                PATTERN_STORE[storage_key] = binary_matrix
                PATTERN_SHAPES[storage_key] = binary_matrix.shape
                print(f"  → Stored as '{storage_key}' (shape: {binary_matrix.shape})")

            except Exception as e:
                print(f"  Error loading {path}: {e}")

    # Store patterns into memory and save cache
    if PATTERN_STORE:
        total_size = int(np.prod(pattern_shape))
        am.__init__(total_size)
        hn.__init__(total_size)

        patterns_am = [(p * 2 - 1).flatten() for p in PATTERN_STORE.values()]
        patterns_hn = [p.flatten() for p in PATTERN_STORE.values()]
        am.store_multiple_patterns(patterns_am)
        hn.store_multiple_patterns(patterns_hn)

        try:
            np.savez(PATTERN_CACHE_FILE,
                     patterns=PATTERN_STORE,
                     shape=pattern_shape,
                     shapes=PATTERN_SHAPES,
                     am_matrix=am.memory,
                     hn_matrix=hn.W)
            print(f"\nSaved pattern + memory cache to '{PATTERN_CACHE_FILE}'")
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
            binary_matrix = np.load(path).flatten()
            print(f"Original vector length: {len(binary_matrix)}")
            if binary_matrix.size >= DOWNSAMPLE_THRESHOLD:
                binary_matrix = downsample_vector(binary_matrix, factor=DOWNSAMPLE_FACTOR)
                print(f"Downsampled vector length: {len(binary_matrix)}")
        else:
            with open(path, 'r') as f:
                content = f.read().strip()
                if all(c in '01' for c in content) and '\n' not in content:
                    binary_array = np.fromiter((int(c) for c in content), dtype=np.int8)
                else:
                    binary_array = np.loadtxt(path).astype(int).flatten()

            binary_matrix = binary_array.flatten()
            print(f"Original vector length: {len(binary_matrix)}")
            if binary_matrix.size >= DOWNSAMPLE_THRESHOLD:
                binary_matrix = downsample_vector(binary_matrix, factor=DOWNSAMPLE_FACTOR)
                print(f"Downsampled vector length: {len(binary_matrix)}")

        if binary_matrix.size != np.prod(pattern_shape):
            print(f"Error: Input shape {binary_matrix.shape} does not match expected {pattern_shape}")
            return

        original = binary_matrix
        noisy = add_noise(binary_matrix, noise_level).flatten()

        am_result = am.recall((noisy * 2 - 1))
        am_bin = np.where(am_result == 1, 1, 0)

        hn_result = hn.recall((noisy * 2 - 1))
        hn_bin = np.where(hn_result == 1, 1, 0)

        acc_am = np.mean(am_bin == original)
        acc_hn = np.mean(hn_bin == original)

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
            return best_name, best_score

        match_am, score_am = find_best_match(am_bin)
        match_hn, score_hn = find_best_match(hn_bin)

        print(f"\nRecall results for: {os.path.basename(path)}")
        print(f"  Accuracy (AM): {acc_am:.2f} | Closest match: {match_am} ({score_am:.2f})")
        print(f"  Accuracy (HN): {acc_hn:.2f} | Closest match: {match_hn} ({score_hn:.2f})")

    except Exception as e:
        print(f"Error during recall: {e}")

def compare_all_patterns(am, hn, noise_level=0.1):
    if not PATTERN_STORE:
        print("No patterns loaded.")
        return

    labels = []
    am_scores = []
    hn_scores = []

    print(f"\nRunning accuracy test on {len(PATTERN_STORE)} patterns (noise: {noise_level})...")
    for name, pattern in PATTERN_STORE.items():
        original = pattern.flatten()
        noisy = add_noise(pattern, noise_level).flatten()

        am_result = am.recall((noisy * 2 - 1))
        am_bin = np.where(am_result == 1, 1, 0)
        acc_am = np.mean(am_bin == original)

        hn_result = hn.recall((noisy * 2 - 1))
        hn_bin = np.where(hn_result == 1, 1, 0)
        acc_hn = np.mean(hn_bin == original)

        labels.append(name)
        am_scores.append(acc_am)
        hn_scores.append(acc_hn)

        print(f"  {name} — AM: {acc_am:.2f}, HN: {acc_hn:.2f}")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 6))
    rects1 = ax.bar(x - width/2, am_scores, width, label='Associative Memory')
    rects2 = ax.bar(x + width/2, hn_scores, width, label='Hopfield Network')

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Pattern Recall Accuracy (Noise = {noise_level})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()

    os.makedirs("charts", exist_ok=True)
    filename = f"charts/accuracy_comparison_noise_{int(noise_level*100)}.png"
    plt.savefig(filename, dpi=300)
    print(f"\nSaved comparison chart to: {filename}")

    plt.show()


if __name__ == "__main__":
    print("Binary Associative Memory & Hopfield Network System\n")
    am = AssociativeMemory(0)
    hn = HopfieldNetwork(0)

    load_all_patterns_from_files(am, hn)

    while True:
        print("\nMenu:")
        print("1. Recall from binary file path")
        print("2. Clear pattern cache and reload")
        print("3. Compare accuracy of all patterns")
        print("4. Exit")

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            path = input("Enter path to binary .txt or .npy file: ").strip()
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
            level = float(input("Enter noise level (0.0 - 1.0): ").strip())
            compare_all_patterns(am, hn, level)
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")