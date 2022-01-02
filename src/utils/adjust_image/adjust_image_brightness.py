from numpy import percentile, zeros
from cv2 import normalize, NORM_MINMAX

def adjust_image_brightness(image):
    max_percentile_pixel = percentile(image, 99)
    min_percentile_pixel = percentile(image, 1)
    
    image[image >= max_percentile_pixel] = max_percentile_pixel
    image[image <= min_percentile_pixel] = min_percentile_pixel

    augmented_image = zeros(image.shape, image.dtype)
    normalize(image, augmented_image, 255*0.1, 255*0.9, NORM_MINMAX)

    return augmented_image