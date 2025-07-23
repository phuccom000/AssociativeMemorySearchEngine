import numpy as np
import os
import warnings
import sys
import matplotlib.pyplot as plt
import math

sys.set_int_max_str_digits(10000)
warnings.filterwarnings("ignore", category=UserWarning)

PATTERN_STORE = {}
PATTERN_SHAPES = {}
PATTERNS_DIR = "patterns"
pattern_shape = None
PATTERN_CACHE_FILE = "pattern_cache.npz"
DOWNSAMPLED_DIR = "patterns_downsampled"
NPY_CONVERTED_DIR = "patterns_npy"
DOWNSAMPLE_THRESHOLD = 10000

def downsample_vector(vector, target_length=DOWNSAMPLE_THRESHOLD):
    original_length = len(vector)
    factor = math.ceil(original_length / target_length)
    padded_length = factor * target_length
    pad_size = padded_length - original_length

    if pad_size > 0:
        vector = np.concatenate([vector, np.zeros(pad_size, dtype=np.int8)])
        print(f"    Padded with {pad_size} zeros to length: {len(vector)}")

    reshaped = vector.reshape(target_length, factor)
    pooled = (np.sum(reshaped, axis=1) >= (factor // 2)).astype(np.int8)

    print(f"    Calculated downsample factor: {factor}")
    print(f"    Downsampled to length: {len(pooled)}")
    return pooled

class AssociativeMemory:
    def __init__(self, size):
        self.size = size
        self.memory = np.zeros((size, size))

    def store_multiple_patterns(self, patterns):
        for pattern in patterns:
            self.memory += np.outer(pattern, pattern)

    def recall(self, input_vector, threshold=0):
        recalled = np.dot(self.memory, input_vector)
        return np.where(recalled > threshold, 1, -1)

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    def store_multiple_patterns(self, patterns):
        for pattern in patterns:
            bipolar = 2 * pattern - 1
            self.W += np.outer(bipolar, bipolar)
        self.W /= len(patterns)
        np.fill_diagonal(self.W, 0)

    def recall(self, pattern, max_iter=100):
        for _ in range(max_iter):
            prev = pattern.copy()
            pattern = np.sign(self.W @ pattern)
            pattern[pattern == 0] = 1
            if np.array_equal(prev, pattern):
                break
        return pattern

def add_noise(pattern, noise_level=0.1):
    mask = np.random.rand(*pattern.shape)
    return np.where(mask < noise_level, 1 - pattern, pattern)

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

                if binary_matrix.size > DOWNSAMPLE_THRESHOLD:
                    binary_matrix = downsample_vector(binary_matrix)
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
    try:
        vec = np.load(path).flatten() if path.endswith(".npy") else np.array([int(c) for c in open(path).read().strip()], dtype=np.int8)
        if vec.size > DOWNSAMPLE_THRESHOLD:
            vec = downsample_vector(vec)
        if vec.shape != pattern_shape:
            print("Pattern shape mismatch.")
            return
        noisy = add_noise(vec, noise_level)

        am_out = am.recall(2 * noisy - 1)
        am_bin = (am_out > 0).astype(np.int8)
        hn_out = hn.recall(2 * noisy - 1)
        hn_bin = (hn_out > 0).astype(np.int8)

        acc_am = np.mean(am_bin == vec)
        acc_hn = np.mean(hn_bin == vec)

        def best_match(recalled):
            best, score = None, 0
            for name, p in PATTERN_STORE.items():
                match = np.mean(p == recalled)
                if match > score:
                    best, score = name, match
            return best, score

        match_am, score_am = best_match(am_bin)
        match_hn, score_hn = best_match(hn_bin)

        print(f"Recall from {os.path.basename(path)} with noise {noise_level}")
        print(f"  AM: Accuracy = {acc_am:.2f}, Closest = {match_am} ({score_am:.2f})")
        print(f"  HN: Accuracy = {acc_hn:.2f}, Closest = {match_hn} ({score_hn:.2f})")

    except Exception as e:
        print(f"Recall error: {e}")

def compare_all_patterns(am, hn, noise_level=0.1):
    if not PATTERN_STORE:
        print("No patterns loaded.")
        return

    labels, acc_am_list, acc_hn_list = [], [], []
    print(f"\nComparing all patterns (noise level: {noise_level}):")

    for name, original in PATTERN_STORE.items():
        noisy = add_noise(original, noise_level)
        am_out = am.recall(2 * noisy - 1)
        hn_out = hn.recall(2 * noisy - 1)

        am_bin = (am_out > 0).astype(np.int8)
        hn_bin = (hn_out > 0).astype(np.int8)

        acc_am = np.mean(am_bin == original)
        acc_hn = np.mean(hn_bin == original)

        labels.append(name)
        acc_am_list.append(acc_am)
        acc_hn_list.append(acc_hn)

        print(f"  {name}: AM = {acc_am:.2f}, HN = {acc_hn:.2f}")

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 6))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, acc_am_list, width, label="Associative Memory")
    ax.bar(x + width/2, acc_hn_list, width, label="Hopfield Network")
    ax.set_title(f"Pattern Recall Accuracy @ {noise_level:.2f} Noise")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    os.makedirs("charts", exist_ok=True)
    fname = f"charts/comparison_noise_{int(noise_level*100)}.png"
    plt.savefig(fname, dpi=300)
    print(f"\nChart saved to: {fname}")
    plt.show()

    # Identify most problematic patterns
    low_am = sorted(zip(labels, acc_am_list), key=lambda x: x[1])[:5]
    low_hn = sorted(zip(labels, acc_hn_list), key=lambda x: x[1])[:5]

    print("\nMost Problematic Patterns:")
    print("  AM:", ", ".join([f"{n} ({s:.2f})" for n, s in low_am]))
    print("  HN:", ", ".join([f"{n} ({s:.2f})" for n, s in low_hn]))

def main_menu():
    am = AssociativeMemory(0)
    hn = HopfieldNetwork(0)
    load_all_patterns_from_files(am, hn)

    while True:
        print("\nMenu:")
        print("1. Recall from .npy or .txt file")
        print("2. Compare accuracy of all patterns")
        print("3. Clear cache and reload patterns")
        print("4. Exit")

        choice = input("Choose option (1-4): ").strip()
        if choice == "1":
            path = input("Enter path to input file: ").strip()
            if not os.path.exists(path):
                print("File not found.")
                continue
            noise = float(input("Noise level (0.0 - 1.0): "))
            recall_from_file(path, am, hn, noise)
        elif choice == "2":
            noise = float(input("Noise level (0.0 - 1.0): "))
            compare_all_patterns(am, hn, noise)
        elif choice == "3":
            if os.path.exists(PATTERN_CACHE_FILE):
                os.remove(PATTERN_CACHE_FILE)
                print("Cache cleared.")
            load_all_patterns_from_files(am, hn)
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main_menu()
