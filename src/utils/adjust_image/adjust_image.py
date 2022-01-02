from .adjust_image_brightness import adjust_image_brightness
from .adjust_image_contrast import adjust_image_contrast
from .normalize_image import normalize_image

def adjust_image(image):
    image = normalize_image(image)
    image = adjust_image_contrast(image)
    image = adjust_image_brightness(image)
    return image