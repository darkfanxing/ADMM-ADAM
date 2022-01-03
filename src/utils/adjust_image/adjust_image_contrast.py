from cv2 import createCLAHE, merge

# Reference: https://blog.csdn.net/qq_43426908/article/details/121221054
def adjust_image_contrast(image):
    clahe = createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    clahe_b = clahe.apply(image[:, :, 0])
    clahe_g = clahe.apply(image[:, :, 1])
    clahe_r = clahe.apply(image[:, :, 2])
    return merge((clahe_r, clahe_g, clahe_b))