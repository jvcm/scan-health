import numpy as np
import cv2
import matplotlib.pyplot as plt

def allow_image(filename, imageExtensions):
    if filename == "":
        print("No filename")
        return False
    elif not "." in filename:
        print("No . in filename")
        return False
    elif not filename.split('.')[-1].upper() in imageExtensions:
        print("Invalid format")
        return False
    else:
        return True

def process_img(img_path, img_dims, batch_size):
    test_data = []

    img = plt.imread(img_path)
    img = cv2.resize(img, (img_dims, img_dims))
    img = np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
        
    test_data = np.array(test_data)
    
    return test_data