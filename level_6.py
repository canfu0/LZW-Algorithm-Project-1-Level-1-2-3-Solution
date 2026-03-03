import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Import the classes we built in previous levels
try:
    from level_2 import ImageLZWCoding
    from level_3 import DiffImageLZWCoding
    from level_4 import ColorImageLZWCoding
    from level_5 import ColorDiffImageLZWCoding
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure level_2.py, level_3.py, level_4.py, and level_5.py are in the same folder.")

class LZWCompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("COMP482 - LZW Image Compression Tool")
        self.root.geometry("800x650")
        self.root.configure(padx=20, pady=20)

        self.file_path = None
        self.image_name = None

        self.setup_ui()

    def setup_ui(self):
        # 1. Header
        header = tk.Label(self.root, text="LZW Image Compression & Decompression", font=("Helvetica", 18, "bold"))
        header.pack(pady=(0, 20))

        # 2. File Selection Frame
        frame_file = tk.Frame(self.root)
        frame_file.pack(fill="x", pady=5)
        
        self.btn_select = tk.Button(frame_file, text="Select Image", command=self.select_image, font=("Helvetica", 12), width=15)
        self.btn_select.pack(side="left", padx=(0, 10))
        
        self.lbl_path = tk.Label(frame_file, text="No file selected", fg="gray", font=("Helvetica", 10))
        self.lbl_path.pack(side="left")

        # 3. Compression Level Selection (Radio Buttons)
        frame_options = tk.LabelFrame(self.root, text="Select Compression Level", font=("Helvetica", 12))
        frame_options.pack(fill="x", pady=15, ipady=5, ipadx=5)

        self.level_var = tk.IntVar(value=2) # Default to Level 2

        tk.Radiobutton(frame_options, text="Level 2: Gray Level Compression", variable=self.level_var, value=2, font=("Helvetica", 11)).pack(anchor="w")
        tk.Radiobutton(frame_options, text="Level 3: Gray Level Differences", variable=self.level_var, value=3, font=("Helvetica", 11)).pack(anchor="w")
        tk.Radiobutton(frame_options, text="Level 4: Color Image (RGB) Compression", variable=self.level_var, value=4, font=("Helvetica", 11)).pack(anchor="w")
        tk.Radiobutton(frame_options, text="Level 5: Color Image Differences", variable=self.level_var, value=5, font=("Helvetica", 11)).pack(anchor="w")

        # 4. Action Buttons
        frame_actions = tk.Frame(self.root)
        frame_actions.pack(pady=10)

        self.btn_compress = tk.Button(frame_actions, text="COMPRESS", command=self.compress_action, font=("Helvetica", 12, "bold"), bg="lightblue", width=15, state="disabled")
        self.btn_compress.grid(row=0, column=0, padx=10)

        self.btn_decompress = tk.Button(frame_actions, text="DECOMPRESS", command=self.decompress_action, font=("Helvetica", 12, "bold"), bg="lightgreen", width=15, state="disabled")
        self.btn_decompress.grid(row=0, column=1, padx=10)

        # 5. Image Display Area
        frame_images = tk.Frame(self.root)
        frame_images.pack(fill="both", expand=True, pady=15)

        # Original Image Placeholder
        frame_orig = tk.Frame(frame_images)
        frame_orig.pack(side="left", expand=True)
        tk.Label(frame_orig, text="Original Image", font=("Helvetica", 11, "bold")).pack()
        self.lbl_orig_img = tk.Label(frame_orig, text="[Image will appear here]", bg="lightgray", width=40, height=15)
        self.lbl_orig_img.pack(pady=5)

        # Decompressed Image Placeholder
        frame_decomp = tk.Frame(frame_images)
        frame_decomp.pack(side="right", expand=True)
        tk.Label(frame_decomp, text="Decompressed Image", font=("Helvetica", 11, "bold")).pack()
        self.lbl_decomp_img = tk.Label(frame_decomp, text="[Image will appear here]", bg="lightgray", width=40, height=15)
        self.lbl_decomp_img.pack(pady=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Bitmap Images", "*.bmp"), ("PNG Images", "*.png"), ("All Files", "*.*")])
        if file_path:
            self.file_path = file_path
            # Extract just the file name without extension
            self.image_name = os.path.splitext(os.path.basename(file_path))[0]
            self.lbl_path.config(text=self.file_path, fg="black")
            
            # Enable buttons
            self.btn_compress.config(state="normal")
            
            # Show the original image in the GUI
            self.display_image(self.file_path, self.lbl_orig_img)
            # Reset decompressed image area
            self.lbl_decomp_img.config(image='', text="[Image will appear here]")

    def compress_action(self):
        if not self.file_path:
            return

        level = self.level_var.get()
        try:
            print(f"\n{'='*40}\nGUI: Starting Compression for Level {level}...\n{'='*40}")
            if level == 2:
                lzw = ImageLZWCoding(self.image_name)
                lzw.compress_image_file(self.file_path)
            elif level == 3:
                lzw = DiffImageLZWCoding(self.image_name)
                lzw.compress_difference_image(self.file_path)
            elif level == 4:
                lzw = ColorImageLZWCoding(self.image_name)
                lzw.compress_color_image(self.file_path)
            elif level == 5:
                lzw = ColorDiffImageLZWCoding(self.image_name)
                lzw.compress_color_diff_image(self.file_path)
            
            messagebox.showinfo("Success", f"Compression completed successfully!\n\nPlease check the terminal console for detailed Entropy and Compression Ratio (CR) statistics.")
            self.btn_decompress.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Compression Error", str(e))

    def decompress_action(self):
        if not self.image_name:
            return

        level = self.level_var.get()
        try:
            print(f"\n{'='*40}\nGUI: Starting Decompression for Level {level}...\n{'='*40}")
            if level == 2:
                lzw = ImageLZWCoding(self.image_name)
                restored_array = lzw.decompress_image_file()
                output_ext = "_decompressed.bmp"
            elif level == 3:
                lzw = DiffImageLZWCoding(self.image_name)
                restored_array = lzw.decompress_difference_image()
                output_ext = "_diff_decompressed.bmp"
            elif level == 4:
                lzw = ColorImageLZWCoding(self.image_name)
                restored_array = lzw.decompress_color_image()
                output_ext = "_color_decompressed.bmp"
            elif level == 5:
                lzw = ColorDiffImageLZWCoding(self.image_name)
                restored_array = lzw.decompress_color_diff_image()
                output_ext = "_color_diff_decompressed.bmp"

            current_directory = os.path.dirname(os.path.realpath(__file__))
            output_file_path = os.path.join(current_directory, f"{self.image_name}{output_ext}")
            
            # Show the decompressed image in the GUI
            self.display_image(output_file_path, self.lbl_decomp_img)
            messagebox.showinfo("Success", f"Decompression completed!\n\nImage saved to:\n{output_file_path}")

        except Exception as e:
            messagebox.showerror("Decompression Error", f"Failed to decompress.\nMake sure you compressed it first!\n\nError: {str(e)}")

    def display_image(self, img_path, label_widget):
        """Helper function to resize and display an image on a Tkinter Label."""
        try:
            img = Image.open(img_path)
            img.thumbnail((300, 300)) # Resize for GUI display to prevent overflow
            photo = ImageTk.PhotoImage(img)
            
            label_widget.config(image=photo, text="")
            label_widget.image = photo # Keep a reference to avoid garbage collection
        except Exception as e:
            print(f"Could not load image for GUI: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LZWCompressionGUI(root)
    root.mainloop()