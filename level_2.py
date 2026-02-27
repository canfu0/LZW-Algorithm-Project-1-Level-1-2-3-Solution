import os
import math
import numpy as np
from PIL import Image
from LZW import LZWCoding

def calculate_entropy(image_array):
    """
    Calculates the entropy of the image: H = - sum(p * log2(p))
    """
    # Flatten the 2D matrix to a 1D array
    flat_array = image_array.flatten()
    # Calculate the frequency of each pixel value (0-255)
    counts = np.bincount(flat_array, minlength=256)
    # Calculate probabilities for non-zero counts
    probabilities = counts[counts > 0] / float(len(flat_array))
    # Apply the entropy formula
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

class ImageLZWCoding(LZWCoding):
    """
    A class that inherits from LZWCoding to implement Gray Level Image Compression (Level 2).
    """
    def __init__(self, filename):
        super().__init__(filename, 'image')
        # Store the original dimensions to reconstruct the image later
        self.num_rows = 0
        self.num_cols = 0

    def compress_image_file(self, image_path):
        print("\n--- LEVEL 2: GRAY LEVEL IMAGE COMPRESSION ---")
        
        # 1. Read the image and convert it to grayscale ('L' mode)
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        self.num_rows, self.num_cols = img_array.shape
        
        # 2. Calculate and print the entropy
        entropy = calculate_entropy(img_array)
        print(f"Image Entropy: {entropy:.4f} bits/pixel")
        
        # 3. Flatten the 2D array to 1D and convert pixel values to characters
        flat_array = img_array.flatten()
        text = "".join([chr(pixel) for pixel in flat_array])
        
        # 4. Perform compression using the inherited methods from LZWCoding
        encoded_text_as_integers = self.encode(text)
        encoded_text = self.int_list_to_binary_string(encoded_text_as_integers)
        encoded_text = self.add_code_length_info(encoded_text)
        padded_encoded_text = self.pad_encoded_data(encoded_text)
        byte_array = self.get_byte_array(padded_encoded_text)
        
        # 5. Write the compressed data to a binary file
        current_directory = os.path.dirname(os.path.realpath(__file__))
        output_file = self.filename + '_compressed.bin'
        output_path = os.path.join(current_directory, output_file)
        
        with open(output_path, 'wb') as out_file:
            out_file.write(bytes(byte_array))
            
        # 6. Calculate and print compression details
        uncompressed_size = len(flat_array) # 1 byte per pixel
        compressed_size = len(byte_array)
        compression_ratio = compressed_size / uncompressed_size
        avg_code_length = (compressed_size * 8) / uncompressed_size
        
        print(f"File successfully compressed into '{output_file}'.")
        print(f"Original Size: {uncompressed_size:,d} bytes")
        print(f"Compressed Size: {compressed_size:,d} bytes")
        print(f"Dictionary Code Length: {self.codelength} bits")
        print(f"Average Code Length: {avg_code_length:.4f} bits/pixel")
        print(f"Compression Ratio (CR): {compression_ratio:.4f}")
        
        return output_path

    def decompress_image_file(self):
        print("\n--- LEVEL 2: GRAY LEVEL IMAGE DECOMPRESSION ---")
        current_directory = os.path.dirname(os.path.realpath(__file__))
        input_file = self.filename + '_compressed.bin'
        input_path = os.path.join(current_directory, input_file)
        
        output_file = self.filename + '_decompressed.bmp'
        output_path = os.path.join(current_directory, output_file)
        
        # 1. Read the compressed binary file
        with open(input_path, 'rb') as in_file:
            compressed_data = in_file.read()
            
        # 2. Convert bytes back to a binary string
        from io import StringIO
        bit_string = StringIO()
        for byte in compressed_data:
            bits = bin(byte)[2:].rjust(8, '0')
            bit_string.write(bits)
        bit_string = bit_string.getvalue()
        
        # 3. Decode the binary string using inherited methods
        bit_string = self.remove_padding(bit_string)
        bit_string = self.extract_code_length_info(bit_string)
        encoded_text = self.binary_string_to_int_list(bit_string)
        decompressed_text = self.decode(encoded_text)
        
        # 4. Convert characters back to integer pixel values (0-255)
        restored_flat_array = np.array([ord(c) for c in decompressed_text], dtype=np.uint8)
        
        # 5. Reshape the 1D array back to its original 2D image dimensions
        restored_img_array = restored_flat_array.reshape((self.num_rows, self.num_cols))
        
        # 6. Save the restored image
        restored_img = Image.fromarray(restored_img_array)
        restored_img.save(output_path)
        print(f"Image successfully decompressed into '{output_file}'.")
        
        return restored_img_array


if __name__ == "__main__":
    image_name = "thumbs_up" 
    image_file_path = image_name + ".bmp" 
    
    if not os.path.exists(image_file_path):
        print(f"ERROR: File '{image_file_path}' not found in the directory.")
    else:
        lzw_image = ImageLZWCoding(image_name)
        
        # Compression
        lzw_image.compress_image_file(image_file_path)
        
        # Decompression
        restored_array = lzw_image.decompress_image_file()
        
        # Verification
        print("\n--- LEVEL 2: VERIFICATION ---")
        original_img = Image.open(image_file_path).convert('L')
        original_array = np.array(original_img)
        
        if np.array_equal(original_array, restored_array):
            print("SUCCESS: The original image and the decompressed image are IDENTICAL.")
        else:
            print("ERROR: Data loss detected during compression/decompression.")