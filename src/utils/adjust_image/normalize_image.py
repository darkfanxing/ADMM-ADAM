from numpy import uint8, float64

def normalize_image(image):
    image = image.astype(float64)
    for channel in range(3):
        min_value = image[..., channel].min()
        max_value = image[..., channel].max()
        if min_value != max_value:
            image[..., channel] -= min_value
            image[..., channel] *= 255 / (max_value - min_value)

    image = image.astype(uint8)
    return image