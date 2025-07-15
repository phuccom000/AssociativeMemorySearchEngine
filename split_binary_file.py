import os

def split_wav_to_txt(input_file_path, output_base_dir):
    """
    Splits a .txt file containing lines like 'idXXXXX/.../XXXXX.wav <binary_string>'
    into individual .txt files, preserving the original directory structure.
    
    Args:
        input_file_path (str): Path to the input .txt file.
        output_base_dir (str): Base directory where output files will be saved.
    """
    with open(input_file_path, 'r') as f:
        for line in f:
            # Split the line into .wav path and binary string
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue

            wav_path, binary_str = parts
            
            # Extract directory structure (e.g., 'id10270/5r0dWxy17C8')
            dir_structure = os.path.dirname(wav_path)
            # Extract filename without extension (e.g., '00003' from '00003.wav')
            filename = os.path.basename(wav_path).replace('.wav', '')
            
            # Create full output directory path
            output_dir = os.path.join(output_base_dir, dir_structure)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output file path
            output_path = os.path.join(output_dir, f"{filename}.txt")
            
            # Write the binary string to the file
            with open(output_path, 'w') as out_file:
                out_file.write(binary_str)
            
            print(f"Created: {output_path}")

# Example usage:
input_file = "processed_embeddings_gray_code_dec8.txt"  # Replace with your input file path
output_base_directory = "patterns voice npy"  # Replace with your desired base output directory
split_wav_to_txt(input_file, output_base_directory)