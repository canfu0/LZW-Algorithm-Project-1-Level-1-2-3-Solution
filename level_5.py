import os
import numpy as np
from PIL import Image
from LZW import LZWCoding

def calculate_entropy(image_array):
    """Calculates the entropy of a given 2D image array or difference matrix."""
    flat_array = image_array.flatten()
    counts = np.bincount(flat_array, minlength=256)
    probabilities = counts[counts > 0] / float(len(flat_array))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

class ColorDiffImageLZWCoding(LZWCoding):
    """
    A class that inherits from LZWCoding to implement Color Image Compression 
    using spatial pixel differences for each RGB channel (Level 5).
    """
    def __init__(self, filename):
        super().__init__(filename, 'image')
        self.num_rows = 0
        self.num_cols = 0
        self.channels = ['R', 'G', 'B']

    def get_difference_image(self, img_array):
        """Computes the difference image to reduce entropy for a single channel."""
        arr = img_array.astype(np.int32)
        diff_arr = np.zeros_like(arr)
        
        # Row-wise differences
        diff_arr[:, 1:] = arr[:, 1:] - arr[:, :-1]
        diff_arr[:, 0] = arr[:, 0] 
        
        # Column-wise differences for the first column
        diff_arr[1:, 0] = arr[1:, 0] - arr[:-1, 0]
        
        # Modulo 256 to map negative values to the 0-255 range
        diff_arr = diff_arr % 256
        return diff_arr.astype(np.uint8)

    def restore_from_difference_image(self, diff_arr):
        """Reconstructs the original channel matrix from its difference matrix."""
        diff_arr = diff_arr.astype(np.int32)
        restored = np.zeros_like(diff_arr)
        
        # Reconstruct the first column
        restored[0, 0] = diff_arr[0, 0]
        for i in range(1, self.num_rows):
            restored[i, 0] = (restored[i-1, 0] + diff_arr[i, 0]) % 256
            
        # Reconstruct the rest of the columns
        for j in range(1, self.num_cols):
            restored[:, j] = (restored[:, j-1] + diff_arr[:, j]) % 256
            
        return restored.astype(np.uint8)

    def compress_color_diff_image(self, image_path):
        print("\n--- LEVEL 5: COLOR IMAGE DIFFERENCE COMPRESSION ---")
        
        # Read the image in RGB mode
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        self.num_rows, self.num_cols, _ = img_array.shape
        print(f"Original Image Dimensions: {self.num_rows}x{self.num_cols} (3 Channels)")
        
        total_uncompressed_size = 0
        total_compressed_size = 0
        
        # Process each color channel independently
        for idx, color in enumerate(self.channels):
            print(f"\nProcessing Channel: {color}")
            
            # 1. Extract the channel
            channel_array = img_array[:, :, idx]
            orig_entropy = calculate_entropy(channel_array)
            print(f"  Original Entropy: {orig_entropy:.4f} bits/pixel")
            
            # 2. Compute the difference matrix
            diff_array = self.get_difference_image(channel_array)
            diff_entropy = calculate_entropy(diff_array)
            print(f"  Difference Entropy: {diff_entropy:.4f} bits/pixel")
            
            # 3. Flatten and stringify
            flat_diff = diff_array.flatten()
            text = "".join([chr(pixel) for pixel in flat_diff])
            
            # 4. LZW Compression
            encoded_text_as_integers = self.encode(text)
            encoded_text = self.int_list_to_binary_string(encoded_text_as_integers)
            encoded_text = self.add_code_length_info(encoded_text)
            padded_encoded_text = self.pad_encoded_data(encoded_text)
            byte_array = self.get_byte_array(padded_encoded_text)
            
            # 5. Save to file
            current_directory = os.path.dirname(os.path.realpath(__file__))
            output_file = f"{self.filename}_{color}_diff_compressed.bin"
            output_path = os.path.join(current_directory, output_file)
            
            with open(output_path, 'wb') as out_file:
                out_file.write(bytes(byte_array))
                
            uncompressed_size = len(flat_diff)
            compressed_size = len(byte_array)
            total_uncompressed_size += uncompressed_size
            total_compressed_size += compressed_size
            
            print(f"  Saved as: '{output_file}'")
            print(f"  Channel CR: {(compressed_size / uncompressed_size):.4f}")

        # Overall Summary
        overall_cr = total_compressed_size / total_uncompressed_size
        print("\n--- LEVEL 5 SUMMARY ---")
        print(f"Total Original Size: {total_uncompressed_size:,d} bytes")
        print(f"Total Compressed Size: {total_compressed_size:,d} bytes")
        print(f"Overall Compression Ratio (CR): {overall_cr:.4f}")

    def decompress_color_diff_image(self):
        print("\n--- LEVEL 5: COLOR IMAGE DIFFERENCE DECOMPRESSION ---")
        current_directory = os.path.dirname(os.path.realpath(__file__))
        
        restored_channels = []
        
        for color in self.channels:
            input_file = f"{self.filename}_{color}_diff_compressed.bin"
            input_path = os.path.join(current_directory, input_file)
            
            print(f"Decompressing and reconstructing Channel: {color}...")
            with open(input_path, 'rb') as in_file:
                compressed_data = in_file.read()
                
            from io import StringIO
            bit_string = StringIO()
            for byte in compressed_data:
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string.write(bits)
            bit_string = bit_string.getvalue()
            
            # Decode LZW
            bit_string = self.remove_padding(bit_string)
            bit_string = self.extract_code_length_info(bit_string)
            encoded_text = self.binary_string_to_int_list(bit_string)
            decompressed_text = self.decode(encoded_text)
            
            # Reshape back to difference matrix
            restored_flat_diff = np.array([ord(c) for c in decompressed_text], dtype=np.uint8)
            restored_diff_array = restored_flat_diff.reshape((self.num_rows, self.num_cols))
            
            # Reconstruct original channel matrix from differences
            restored_channel_array = self.restore_from_difference_image(restored_diff_array)
            restored_channels.append(restored_channel_array)
            
        # Stack R, G, B channels
        restored_img_array = np.dstack(restored_channels)
        
        output_file = f"{self.filename}_color_diff_decompressed.bmp"
        output_path = os.path.join(current_directory, output_file)
        
        restored_img = Image.fromarray(restored_img_array, 'RGB')
        restored_img.save(output_path)
        print(f"\nColor image successfully reconstructed and saved as '{output_file}'.")
        
        return restored_img_array


if __name__ == "__main__":
    image_name = "thumbs_up" 
    image_file_path = image_name + ".bmp" 
    
    if not os.path.exists(image_file_path):
        print(f"ERROR: File '{image_file_path}' not found in the directory.")
    else:
        lzw_color_diff = ColorDiffImageLZWCoding(image_name)
        
        # Compress
        lzw_color_diff.compress_color_diff_image(image_file_path)
        
        # Decompress
        restored_color_array = lzw_color_diff.decompress_color_diff_image()
        
        # Verification
        print("\n--- LEVEL 5: VERIFICATION ---")
        original_img = Image.open(image_file_path).convert('RGB')
        original_color_array = np.array(original_img)
        
        if np.array_equal(original_color_array, restored_color_array):
            print("SUCCESS: The original color image and the reconstructed color image are IDENTICAL.")
        else:
            print("ERROR: Data loss detected.")