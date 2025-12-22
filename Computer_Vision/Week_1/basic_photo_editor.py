import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy

#----------------CONFIG---------
WIDTH = 1200
HEIGHT = 600
BTN_WIDTH = 30
WINDOW_W = 300 # image window width
WINDOW_H = 300 # image window height
# ----------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Photo Editor by Harsh Singh")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)
        self.top_frame = tk.Frame(self.root, bg = "gray")
        self.top_frame.pack(fill = 'x', pady = 10)

        self.main_frame = tk.Frame(self.root, bg = "black")
        self.main_frame.pack(expand = True)
        self.running = True
        self.image_uploaded = False
        self.image_edited = False
        self.original_image_label = tk.Label(self.main_frame, bg = "lightgray") # here tk interpret it as text size not pixel size, so we ignore resizing label for now
        self.download_image_label = tk.Label(self.main_frame, bg = "lightgray")
        self.original_image = None
        self.download_image = None
        self.create_buttons()
        self.download_count = 1
        self.blurring_kernel_size = (10, 10)
        # parameters for canny edges
        self.t_lower = 100
        self.t_upper = 200
        self.aperture_size = 5 # must be odd number(because then only we would have midpoint of sliding matrix), and higher for shaper edges
        
    def upload_image(self):
        fileTypes = [("Image File", "*.png *.jpeg")]
        path = filedialog.askopenfilename(filetypes=fileTypes)
        if(len(path)):
            self.image_uploaded = True
            img = Image.open(path)
            self.original_image = cv2.imread(path)
            img = img.resize((WINDOW_W, WINDOW_H), Image.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)

            self.original_image_label.config(image = photo)
            self.original_image_label.image = photo # keeping the reference

    def download_the_image(self):
        if not (self.image_edited): # or if downloaded image is None
            return 
        img = Image.fromarray(cv2.cvtColor(self.download_image, cv2.COLOR_BGR2RGB))
        file_name = filedialog.asksaveasfilename()
        img.save(file_name)
        self.download_count += 1

    def create_canny_edge(self):
        if not self.image_uploaded:
            return 
        self.image_edited = True
        #convert the numpy array(from original image) to grayscale and then apply canny edge to get download image
        edge = cv2.Canny(cv2.cvtColor(cv2.resize(self.original_image, (WINDOW_W, WINDOW_H)), cv2.COLOR_BGR2GRAY), self.t_lower, self.t_upper, self.aperture_size)

        # Numpy Image -> PIL Image
        img_pil = Image.fromarray(edge)
        # PIL Image -> ImageTk
        photo = ImageTk.PhotoImage(img_pil)
        self.download_image_label.config(image = photo)
        self.download_image_label.image = photo # for future reference
        
    def blur_image(self):
        if not self.image_uploaded:
            return 

        self.image_edited = True
        # blur the image starting from original image
        self.download_image = cv2.blur(cv2.resize(self.original_image, (WINDOW_W, WINDOW_H)), self.blurring_kernel_size)

        # convert the numpy image to PIL image( after converting BGR to RGB)
        img_pil = Image.fromarray(cv2.cvtColor(self.download_image, cv2.COLOR_BGR2RGB))
        # get photo of img to be stored in memory for future reference
        photo = ImageTk.PhotoImage(img_pil)

        # assign reference of photo to download_image_label
        self.download_image_label.config(image = photo)
        # save the photo in memory 
        self.download_image_label.image = photo
        
    def gray_the_image(self):
        if not self.image_uploaded:
            return
        self.image_edited = True
        # processing always starts from original image

        # update the download image as gray image starting from original image not from downloaded image
        self.download_image = cv2.resize(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), (WINDOW_W, WINDOW_H)) 
        # no need to convert bgr to rgb as in gray scale all values are equal

        # convert the numpy array to PIL image
        img_pil = Image.fromarray(self.download_image)
        # store a copy of PIL image as photo for future reference
        photo = ImageTk.PhotoImage(img_pil)

        # display the photo in download_image_label( passed by reference)
        self.download_image_label.config(image = photo) # label config , we give photo image
        self.download_image_label.image = photo # to keep the reference of photo in computer memory
    
    def create_buttons(self):
        upload_btn = tk.Button(self.top_frame, text="UPLOAD", width=BTN_WIDTH, command=self.upload_image)
        blur_btn = tk.Button(self.top_frame, text="BLUR", width=BTN_WIDTH, command=self.blur_image)
        gray_btn = tk.Button(self.top_frame, text="GRAY", width=BTN_WIDTH, command=self.gray_the_image)
        edge_btn = tk.Button(self.top_frame, text="EDGE", width=BTN_WIDTH, command=self.create_canny_edge)
        download_btn = tk.Button(self.top_frame, text="DOWNLOAD", width=BTN_WIDTH, command=self.download_the_image)
        
        upload_btn.grid(row=0, column=0, padx=10)
        blur_btn.grid(row=0, column=1, padx=10)
        gray_btn.grid(row=0, column=2, padx=10)
        edge_btn.grid(row=0, column=3, padx=10)
        download_btn.grid(row=0, column=4, padx=10)
        
        self.original_image_label.grid(row=0, column=0, padx=100, pady=50)
        self.download_image_label.grid(row=0, column=1, padx=100, pady=50)

# ----------RUN APP---------------------
if __name__ == "__main__":
    root =tk.Tk()
    app = App(root)
    root.mainloop()
