import os
import numpy as np

def convert_txt_to_npy_recursive(input_dir='patterns', output_dir='patterns_npy'):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            txt_path = os.path.join(root, file)
            relative_path = os.path.relpath(txt_path, input_dir)
            npy_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".npy")

            os.makedirs(os.path.dirname(npy_path), exist_ok=True)

            try:
                # Try loading as multiline matrix
                try:
                    data = np.loadtxt(txt_path).astype(int)
                except:
                    # Fall back to single-line binary string
                    with open(txt_path, 'r') as f:
                        content = f.read().strip()
                        if all(c in '01' for c in content):
                            data = np.array([int(c) for c in content])
                        else:
                            raise ValueError("Invalid characters in single-line binary.")

                np.save(npy_path, data)
                print(f"Converted {txt_path} -> {npy_path}")

            except Exception as e:
                print(f"Failed to convert {txt_path}: {e}")

if __name__ == "__main__":
    convert_txt_to_npy_recursive()
