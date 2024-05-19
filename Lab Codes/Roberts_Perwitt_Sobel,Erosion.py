import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label, Tk, Toplevel
from PIL import Image, ImageTk

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

    def load_default_image(self):
        path = 'images/OIP.jpeg'
        self.original_image = cv2.imread(path)
        if self.original_image is not None:
            self.update_image(self.original_image)
        else:
            print(f"Error: Unable to load image from {path}")

    def load_image(self):
        path = 'images/OIP.jpeg'
        self.original_image = cv2.imread(path)
        if self.original_image is not None:
            self.update_image(self.original_image)
        else:
            print(f"Error: Unable to load image from {path}")

    def update_image(self, image):
        max_width = 600
        max_height = 400
        image = self.resize_image(image, max_width, max_height)

        if len(image.shape) == 2:  # If the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # If the image is BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.image_label.configure(image=image)
        self.image_label.image = image

    def show_image_in_new_window(self, image):
        new_window = Toplevel(self.master)
        new_window.title('Processed Image')
        
        max_width = 600
        max_height = 400
        image = self.resize_image(image, max_width, max_height)

        if len(image.shape) == 2:  # If the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # If the image is BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        new_image_label = Label(new_window, image=image)
        new_image_label.image = image
        new_image_label.pack()

    def resize_image(self, image, max_width, max_height):
        height, width = image.shape[:2]
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = cv2.resize(image, (new_width, new_height))
        return image

    def add_buttons(self):
        self.add_button('Roberts Edge Detector', self.apply_roberts_edge_detector)
        self.add_button('Prewitt Edge Detector', self.apply_prewitt_edge_detector)
        self.add_button('Sobel Edge Detector', self.apply_sobel_edge_detector)
        self.add_button('Erosion', self.apply_erosion)

    def add_button(self, text, command):
        button = Button(self.master, text=text, command=command, font=("Arial", 10))
        button.pack(pady=5, padx=10)

    def apply_roberts_edge_detector(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0], [0, -1]], dtype=int)
        kernely = np.array([[0, 1], [-1, 0]], dtype=int)
        roberts_x = cv2.filter2D(gray_image, cv2.CV_64F, kernelx)
        roberts_y = cv2.filter2D(gray_image, cv2.CV_64F, kernely)
        roberts_image = np.sqrt(roberts_x**2 + roberts_y**2)
        roberts_image = cv2.normalize(roberts_image, None, 0, 255, cv2.NORM_MINMAX)
        roberts_image = roberts_image.astype(np.uint8)
        self.show_image_in_new_window(roberts_image)

    def apply_prewitt_edge_detector(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        prewitt_x = cv2.filter2D(gray_image, cv2.CV_64F, kernelx)
        prewitt_y = cv2.filter2D(gray_image, cv2.CV_64F, kernely)
        prewitt_image = np.sqrt(prewitt_x**2 + prewitt_y**2)
        prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX)
        prewitt_image = prewitt_image.astype(np.uint8)
        self.show_image_in_new_window(prewitt_image)

    def apply_sobel_edge_detector(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_image = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX)
        sobel_image = sobel_image.astype(np.uint8)
        self.show_image_in_new_window(sobel_image)

    def apply_erosion(self):
        kernel_size = 5  # Fixed kernel size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion_image = cv2.erode(self.original_image, kernel, iterations=1)
        self.show_image_in_new_window(erosion_image)


if __name__ == '__main__':
    root = Tk()
    app = ImageProcessingApp(root)
    root.geometry("800x600")
    root.mainloop()
