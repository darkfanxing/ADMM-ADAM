from cv2 import createCLAHE, merge

def adjust_image_contrast(image):
    clahe = createCLAHE(clipLimit=1, tileGridSize=(8, 8))

    clahe_B = clahe.apply(image[:, :, 0])
    clahe_G = clahe.apply(image[:, :, 1])
    clahe_R = clahe.apply(image[:, :, 2])

    return merge((clahe_R, clahe_G, clahe_B))