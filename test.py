from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# Initializing Tkinter root
root = tk.Tk()
root.withdraw()

# Prompt the user to select an image file
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

# Check if a file was selected
if not image_path: 
    print("No file selected. Exiting.")

else: 
    # Initialize the YOLO model with your custom model weights

    model_weights = "runs/detect/train/weights/best.pt"

    model = YOLO(model_weights)

    result = model(image_path, save=True)


