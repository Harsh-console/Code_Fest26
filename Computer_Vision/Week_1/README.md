We split whole display into two frames: 1. buttons frame, and 2. displaying image frame

i have three types of buttons.
1. upload
2. donwload
3. drop down list for chagning effects
4. in changing effects first i have blur(
5. blur (for now blurrs the whole image, for simplicity, to be upgraded later)
6. then i have rgb to gray
7. then we have edge detection
8. i will also add other features later, so i would kepp some space for other buttons
9. and finally there will be two big boxes down showing images before and after depeneding upon wheather image is upload and wheather images in affects.

Buttons in tkinter:
1. btn = tk.Button(root, text=" harsh singh", command = function to call)
   btn.pack()
   this creates button once, and when ever its clicked while mainloop is running , then it calls the command function. so i could make a function create_button only once to create the buttons(i hope there is no scope related errors.).
2. I could also add drop down list for buttons in edditing as well:
3. (lets ignore drop down list for now)

VIDEO DEMO :-


https://github.com/user-attachments/assets/e65f2f30-94f5-4ce6-aa67-1ea42a90f08d



website used from now(i forgot to add this):
1. https://www.geeksforgeeks.org/python/browse-upload-display-image-in-tkinter/
2. https://www.geeksforgeeks.org/python/dropdown-menus-tkinter/
3. https://www.geeksforgeeks.org/python/image-resizing-using-opencv-python/
4. https://www.geeksforgeeks.org/python/convert-a-numpy-array-to-an-image/
5. https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
6. https://www.geeksforgeeks.org/python/python-opencv-cv2-blur-method/
7. https://www.geeksforgeeks.org/python/python-opencv-canny-function/
8. https://docs.python.org/3/library/dialog.html

internally we do :
1.OpenCV (BGR / Gray) → NumPy
NumPy → PIL.Image
PIL.Image → ImageTk.PhotoImage

requirement:

##  required - pip install pillow

1. resized_image = cv2.resize(src, dsize, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
2. cv2.blur(src, ksize[, dst[, anchor[, borderType]]])
3. Syntax: cv2.Canny(image, T_lower, T_upper, aperture_size, L2Gradient)


