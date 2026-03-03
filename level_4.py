import os
import numpy as np
from PIL import Image
from LZW import LZWCoding

def calculate_entropy(image_array):
    """Calculates the entropy of a given 2D image array."""
    flat_array = image_array.flatten()
    counts = np.bincount(flat_array, minlength=256)
    probabilities = counts[counts > 0] / float(len(flat_array))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

class ColorImageLZWCoding(LZWCoding):
    """
    A class that inherits from LZWCoding to implement Color Image Compression (Level 4).
    It processes R, G, and B channels separately as instructed.
    """
    def __init__(self, filename):
        super().__init__(filename, 'image')
        self.num_rows = 0
        self.num_cols = 0
        self.channels = ['R', 'G', 'B']

    def compress_color_image(self, image_path):
        print("\n--- LEVEL 4: COLOR IMAGE COMPRESSION (RGB) ---")
        
        # 1. Read the image in RGB mode
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        self.num_rows, self.num_cols, _ = img_array.shape
        print(f"Original Image Dimensions: {self.num_rows}x{self.num_cols} (3 Channels)")
        
        total_uncompressed_size = 0
        total_compressed_size = 0
        
        # 2. Process each color channel (R=0, G=1, B=2) independently
        for idx, color in enumerate(self.channels):
            print(f"\nProcessing Channel: {color}")
            
            # Extract the specific 2D color layer
            channel_array = img_array[:, :, idx]
            
            # Calculate and print entropy for this specific channel
            entropy = calculate_entropy(channel_array)
            print(f"  {color} Channel Entropy: {entropy:.4f} bits/pixel")
            
            # Flatten and convert to string for LZW
            flat_array = channel_array.flatten()
            text = "".join([chr(pixel) for pixel in flat_array])
            
            # Perform LZW compression using inherited methods
            encoded_text_as_integers = self.encode(text)
            encoded_text = self.int_list_to_binary_string(encoded_text_as_integers)
            encoded_text = self.add_code_length_info(encoded_text)
            padded_encoded_text = self.pad_encoded_data(encoded_text)
            byte_array = self.get_byte_array(padded_encoded_text)
            
            # Save the compressed binary file for this specific channel
            current_directory = os.path.dirname(os.path.realpath(__file__))
            output_file = f"{self.filename}_{color}_compressed.bin"
            output_path = os.path.join(current_directory, output_file)
            
            with open(output_path, 'wb') as out_file:
                out_file.write(bytes(byte_array))
                
            # Accumulate sizes for final ratio calculation
            uncompressed_size = len(flat_array)
            compressed_size = len(byte_array)
            total_uncompressed_size += uncompressed_size
            total_compressed_size += compressed_size
            
            print(f"  Saved as: '{output_file}'")
            print(f"  Channel CR: {(compressed_size / uncompressed_size):.4f}")

        # 3. Print overall compression statistics
        overall_cr = total_compressed_size / total_uncompressed_size
        overall_avg_length = (total_compressed_size * 8) / (self.num_rows * self.num_cols * 3)
        print("\n--- LEVEL 4 SUMMARY ---")
        print(f"Total Original Size: {total_uncompressed_size:,d} bytes")
        print(f"Total Compressed Size: {total_compressed_size:,d} bytes")
        print(f"Overall Average Code Length: {overall_avg_length:.4f} bits/pixel")
        print(f"Overall Compression Ratio (CR): {overall_cr:.4f}")

    def decompress_color_image(self):
        print("\n--- LEVEL 4: COLOR IMAGE DECOMPRESSION ---")
        current_directory = os.path.dirname(os.path.realpath(__file__))
        
        restored_channels = []
        
        # 1. Read and decode each channel file sequentially
        for color in self.channels:
            input_file = f"{self.filename}_{color}_compressed.bin"
            input_path = os.path.join(current_directory, input_file)
            
            print(f"Decompressing Channel: {color} from '{input_file}'...")
            with open(input_path, 'rb') as in_file:
                compressed_data = in_file.read()
                
            from io import StringIO
            bit_string = StringIO()
            for byte in compressed_data:
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string.write(bits)
            bit_string = bit_string.getvalue()
            
            # Decode using inherited methods
            bit_string = self.remove_padding(bit_string)
            bit_string = self.extract_code_length_info(bit_string)
            encoded_text = self.binary_string_to_int_list(bit_string)
            decompressed_text = self.decode(encoded_text)
            
            # Reconstruct the 2D matrix for this color channel
            restored_flat_array = np.array([ord(c) for c in decompressed_text], dtype=np.uint8)
            restored_channel_array = restored_flat_array.reshape((self.num_rows, self.num_cols))
            
            restored_channels.append(restored_channel_array)
            
        # 2. Stack the R, G, and B 2D matrices back into a 3D color image array
        restored_img_array = np.dstack(restored_channels)
        
        # 3. Save the final reconstructed color image
        output_file = f"{self.filename}_color_decompressed.bmp"
        output_path = os.path.join(current_directory, output_file)
        
        restored_img = Image.fromarray(restored_img_array, 'RGB')
        restored_img.save(output_path)
        print(f"\nColor image successfully reconstructed and saved as '{output_file}'.")
        
        return restored_img_array


if __name__ == "__main__":
    # Test resmi olarak zip'ten cikan thumbs_up.bmp'yi kullanacagiz
    # Eger elinde orijinal renkli baska bir resim varsa adini buraya yazabilirsin.
    image_name = "thumbs_up" 
    image_file_path = image_name + ".bmp" 
    
    if not os.path.exists(image_file_path):
        print(f"ERROR: File '{image_file_path}' not found in the directory.")
    else:
        lzw_color = ColorImageLZWCoding(image_name)
        
        # Sıkıştırma (Compression)
        lzw_color.compress_color_image(image_file_path)
        
        # Geri Açma (Decompression)
        restored_color_array = lzw_color.decompress_color_image()
        
        # Dogrulama (Verification)
        print("\n--- LEVEL 4: VERIFICATION ---")
        original_img = Image.open(image_file_path).convert('RGB')
        original_color_array = np.array(original_img)
        
        if np.array_equal(original_color_array, restored_color_array):
            print("SUCCESS: The original color image and the reconstructed color image are IDENTICAL.")
        else:
            print("ERROR: Data loss detected during color image compression/decompression.")