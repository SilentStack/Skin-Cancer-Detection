import cv2

def load_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128)) 
