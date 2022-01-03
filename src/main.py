from model import ADMMADAM
from utils import load_data, adjust_image
from numpy import expand_dims
import cv2

if __name__ == "__main__":
    IMAGE_REFERENCE_3D, IMAGE_REFERENCE_MASK_3D, IMAGE_DL_3D = load_data()
    IMAGE_CORRUPTED_3D = IMAGE_REFERENCE_3D * IMAGE_REFERENCE_MASK_3D
    
    admm_adam_framework = ADMMADAM(IMAGE_REFERENCE_3D, IMAGE_REFERENCE_MASK_3D, IMAGE_CORRUPTED_3D, IMAGE_DL_3D)
    image_recovery = admm_adam_framework.restore_image()

    rgb_band = [17, 7, 1]
    cv2.imshow("recovery_image", adjust_image(image_recovery[:, :, rgb_band]))
    cv2.imshow("reference_image", adjust_image(IMAGE_REFERENCE_3D[:, :, rgb_band]))
    cv2.imshow(
        "corrupted_image",
        adjust_image(IMAGE_REFERENCE_3D[:, :, rgb_band])
            * expand_dims(IMAGE_REFERENCE_MASK_3D[:, :, rgb_band[0]], axis=2)
            * expand_dims(IMAGE_REFERENCE_MASK_3D[:, :, rgb_band[1]], axis=2)
            * expand_dims(IMAGE_REFERENCE_MASK_3D[:, :, rgb_band[2]], axis=2)
    )
    cv2.imshow("GAN_with_ADAM_image", adjust_image(IMAGE_DL_3D[:, :, rgb_band]))
    cv2.waitKey(0)