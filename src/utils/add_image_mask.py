from numpy import expand_dims

def add_image_mask(image, image_mask):
    image_with_mask = image.copy()
    for channel in range(image.shape[2]):
        image_with_mask *= expand_dims(image_mask[:, :, channel], axis=2)

    return image_with_mask