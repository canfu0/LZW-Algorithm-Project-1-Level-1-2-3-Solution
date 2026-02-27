import os
import numpy as np
from PIL import Image
from LZW import LZWCoding

def calculate_entropy(image_array):
    """Calculates the entropy of the given image or difference matrix."""
    flat_array = image_array.flatten()
    counts = np.bincount(flat_array, minlength=256)
    probabilities = counts[counts > 0] / float(len(flat_array))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

class DiffImageLZWCoding(LZWCoding):
    """
    A class that inherits from LZWCoding to implement Gray Level Image Compression 
    using pixel differences (Level 3).
    """
    def __init__(self, filename):
        super().__init__(filename, 'image')
        self.num_rows = 0
        self.num_cols = 0

    def get_difference_image(self, img_array):
        """
        Computes the difference image to reduce entropy.
        Row-wise differences for all columns, and column-wise differences for the first column.
        """
        # Convert to int32 to prevent overflow during subtraction
        arr = img_array.astype(np.int32)
        diff_arr = np.zeros_like(arr)
        
        # 1. Row-wise differences: subtract the left pixel from the current pixel
        diff_arr[:, 1:] = arr[:, 1:] - arr[:, :-1]
        
        # Keep the first column as is temporarily
        diff_arr[:, 0] = arr[:, 0] 
        
        # 2. Column-wise differences for the first column: subtract the top pixel
        diff_arr[1:, 0] = arr[1:, 0] - arr[:-1, 0]
        
        # Use modulo 256 to map negative differences back to the 0-255 range (lossless)
        diff_arr = diff_arr % 256
        return diff_arr.astype(np.uint8)

    def restore_from_difference_image(self, diff_arr):
        """
        Reconstructs the original image matrix from the difference matrix.
        """
        diff_arr = diff_arr.astype(np.int32)
        restored = np.zeros_like(diff_arr)
        
        # 1. Reconstruct the first column from top to bottom
        restored[0, 0] = diff_arr[0, 0]
        for i in range(1, self.num_rows):
            restored[i, 0] = (restored[i-1, 0] + diff_arr[i, 0]) % 256
            
        # 2. Reconstruct the remaining columns from left to right
        for j in range(1, self.num_cols):
            restored[:, j] = (restored[:, j-1] + diff_arr[:, j]) % 256
            
        return restored.astype(np.uint8)

    def compress_difference_image(self, image_path):
        print("\n--- LEVEL 3: IMAGE COMPRESSION (DIFFERENCES) ---")
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        self.num_rows, self.num_cols = img_array.shape
        
        # Entropy of the original image
        orig_entropy = calculate_entropy(img_array)
        print(f"Original Image Entropy: {orig_entropy:.4f} bits/pixel")
        
        # Compute the difference image and its entropy
        diff_array = self.get_difference_image(img_array)
        diff_entropy = calculate_entropy(diff_array)
        print(f"Difference Image Entropy: {diff_entropy:.4f} bits/pixel")
        
        # Flatten the difference matrix and convert to a character string
        flat_diff = diff_array.flatten()
        text = "".join([chr(pixel) for pixel in flat_diff])
        
        # Perform LZW compression
        encoded_text_as_integers = self.encode(text)
        encoded_text = self.int_list_to_binary_string(encoded_text_as_integers)
        encoded_text = self.add_code_length_info(encoded_text)
        padded_encoded_text = self.pad_encoded_data(encoded_text)
        byte_array = self.get_byte_array(padded_encoded_text)
        
        # Write to a binary file
        current_directory = os.path.dirname(os.path.realpath(__file__))
        output_file = self.filename + '_diff_compressed.bin'
        output_path = os.path.join(current_directory, output_file)
        
        with open(output_path, 'wb') as out_file:
            out_file.write(bytes(byte_array))
            
        uncompressed_size = len(flat_diff)
        compressed_size = len(byte_array)
        
        print(f"\nFile successfully compressed into '{output_file}'.")
        print(f"Original Size: {uncompressed_size:,d} bytes")
        print(f"Compressed Size: {compressed_size:,d} bytes")
        print(f"Compression Ratio (CR): {(compressed_size / uncompressed_size):.4f}")
        return output_path

    def decompress_difference_image(self):
        print("\n--- LEVEL 3: DIFFERENCE IMAGE DECOMPRESSION ---")
        current_directory = os.path.dirname(os.path.realpath(__file__))
        input_file = self.filename + '_diff_compressed.bin'
        input_path = os.path.join(current_directory, input_file)
        output_file = self.filename + '_diff_decompressed.bmp'
        output_path = os.path.join(current_directory, output_file)
        
        with open(input_path, 'rb') as in_file:
            compressed_data = in_file.read()
            
        from io import StringIO
        bit_string = StringIO()
        for byte in compressed_data:
            bits = bin(byte)[2:].rjust(8, '0')
            bit_string.write(bits)
        bit_string = bit_string.getvalue()
        
        bit_string = self.remove_padding(bit_string)
        bit_string = self.extract_code_length_info(bit_string)
        encoded_text = self.binary_string_to_int_list(bit_string)
        decompressed_text = self.decode(encoded_text)
        
        # Convert characters back to numeric values to reconstruct the 2D difference matrix
        restored_flat_diff = np.array([ord(c) for c in decompressed_text], dtype=np.uint8)
        restored_diff_array = restored_flat_diff.reshape((self.num_rows, self.num_cols))
        
        # Reconstruct the original matrix from the difference matrix
        restored_img_array = self.restore_from_difference_image(restored_diff_array)
        
        restored_img = Image.fromarray(restored_img_array)
        restored_img.save(output_path)
        print(f"Image successfully reconstructed from differences and saved as '{output_file}'.")
        return restored_img_array


if __name__ == "__main__":
    image_name = "thumbs_up" 
    image_file_path = image_name + ".bmp" 
    
    if not os.path.exists(image_file_path):
        print(f"ERROR: File '{image_file_path}' not found in the directory.")
    else:
        lzw_diff = DiffImageLZWCoding(image_name)
        
        # Execute Compression
        lzw_diff.compress_difference_image(image_file_path)
        
        # Execute Decompression
        restored_array = lzw_diff.decompress_difference_image()
        
        # Verification
        print("\n--- LEVEL 3: VERIFICATION ---")
        original_img = Image.open(image_file_path).convert('L')
        original_array = np.array(original_img)
        
        if np.array_equal(original_array, restored_array):
            print("SUCCESS: The original image and the reconstructed image are IDENTICAL.")
        else:
            print("ERROR: Data loss detected during the difference reconstruction process.")