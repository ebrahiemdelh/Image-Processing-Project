import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label, Tk
from PIL import Image, ImageTk

class OutputWindow:
    def __init__(self, master, image):
        self.master = master
        self.master.title('Processed Image')

        self.image_label = Label(master)
        self.image_label.pack()

        self.photo_image = self.convert_image(image)

        self.image_label.configure(image=self.photo_image)

    def convert_image(self, image):
        max_width = 600
        max_height = 400
        image = self.resize_image(image, max_width, max_height)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo_image = ImageTk.PhotoImage(image)

        return photo_image

    def resize_image(self, image, max_width, max_height):
        height, width, _ = image.shape
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = cv2.resize(image, (new_width, new_height))
        return image

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Image Processing App')

        self.image_label = Label(master)
        self.image_label.pack()

        self.load_image_button = Button(master, text='Load Image', command=self.load_image)
        self.load_image_button.pack()

        self.add_buttons()

        self.load_default_image()

        self.output_window = None  # Initialize OutputWindow instance as None

    def load_default_image(self):
        path = 'images/OIP.jpeg'
        self.original_image = cv2.imread(path)
        self.update_image(self.original_image)

    def load_image(self):
        path = 'images/OIP.jpeg'
        self.original_image = cv2.imread(path)
        self.update_image(self.original_image)

    def update_image(self, image):
        max_width = 600
        max_height = 400
        image = self.resize_image(image, max_width, max_height)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.image_label.configure(image=image)
        self.image_label.image = image

    def resize_image(self, image, max_width, max_height):
        height, width, _ = image.shape
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = cv2.resize(image, (new_width, new_height))
        return image

    def add_buttons(self):
        self.add_button('LPF', self.apply_lpf)
        self.add_button('HPF', self.apply_hpf)
        self.add_button('Mean Filter', self.apply_mean_filter)
        self.add_button('Median Filter', self.apply_median_filter)

    def add_button(self, text, command):
        button = Button(self.master, text=text, command=command, font=("Arial", 10))
        button.pack(pady=5, padx=10)

    def apply_lpf(self):
        lpf_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        self.output_window = OutputWindow(tk.Toplevel(), lpf_image)  # Create OutputWindow instance with Toplevel widget

    def apply_hpf(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        hpf_image = cv2.subtract(gray_image, cv2.GaussianBlur(gray_image, (5, 5), 0))
        self.output_window = OutputWindow(tk.Toplevel(), cv2.cvtColor(hpf_image, cv2.COLOR_GRAY2BGR))

    def apply_mean_filter(self):
        mean_image = cv2.blur(self.original_image, (5, 5))
        self.output_window = OutputWindow(tk.Toplevel(), mean_image)

    def apply_median_filter(self):
        median_image = cv2.medianBlur(self.original_image, 5)
        self.output_window = OutputWindow(tk.Toplevel(), median_image)

if __name__ == '__main__':
    root = Tk()
    app = ImageProcessingApp(root)

    # Set the window size
    root.geometry("800x600")  # Adjust the width and height as per your preference

    root.mainloop()
