import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label, Tk, Toplevel
from PIL import Image, ImageTk

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Image Processing App')

        self.original_image_label = Label(master)
        self.original_image_label.pack()

        self.load_image_button = Button(master, text='Load Image', command=self.load_image)
        self.load_image_button.pack()

        self.add_buttons()

        self.load_default_image()

    def load_default_image(self):
        path = 'images/OIP.jpeg'
        self.original_image = cv2.imread(path)
        if self.original_image is not None:
            self.update_original_image(self.original_image)
        else:
            print(f"Error: Unable to load image from {path}")

    def load_image(self):
        path = 'images/OIP.jpeg'
        self.original_image = cv2.imread(path)
        if self.original_image is not None:
            self.update_original_image(self.original_image)
        else:
            print(f"Error: Unable to load image from {path}")

    def update_original_image(self, image):
        max_width = 600
        max_height = 400
        image = self.resize_image(image, max_width, max_height)

        if len(image.shape) == 2:  # If the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # If the image is BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.original_image_label.configure(image=image)
        self.original_image_label.image = image

    def update_processed_image(self, image):
        max_width = 600
        max_height = 400
        image = self.resize_image(image, max_width, max_height)

        if len(image.shape) == 2:  # If the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # If the image is BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Open a new window to display the processed image
        new_window = Toplevel(self.master)
        new_window.title('Processed Image')

        processed_image_label = Label(new_window, image=image)
        processed_image_label.image = image
        processed_image_label.pack()

    def resize_image(self, image, max_width, max_height):
        height, width = image.shape[:2]
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = cv2.resize(image, (new_width, new_height))
        return image

    def add_buttons(self):
        self.add_button('Dilation', self.apply_dilation)
        self.add_button('Open', self.apply_open)
        self.add_button('Close', self.apply_close)
        self.add_button('Hough', self.apply_hough_circle_transform)

    def add_button(self, text, command):
        button = Button(self.master, text=text, command=command, font=("Arial", 10))
        button.pack(pady=5, padx=10)

    def apply_dilation(self):
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation_image = cv2.dilate(self.original_image, kernel, iterations=1)
        self.update_processed_image(dilation_image)

    def apply_open(self):
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        open_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
        self.update_processed_image(open_image)

    def apply_close(self):
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        close_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
        self.update_processed_image(close_image)

    def apply_hough_circle_transform(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            hough_image = self.original_image.copy()
            for i in circles[0, :]:
                cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the outer circle
                cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)       # Draw the center of the circle
            self.update_processed_image(hough_image)


if __name__ == '__main__':
    root = Tk()
    app = ImageProcessingApp(root)
    root.geometry("800x600")
    root.mainloop()
