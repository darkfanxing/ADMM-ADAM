from utils import ADMMADAM, load_data
from scipy.io import savemat


if __name__ == "__main__":
    IMAGE_REFERENCE, IMAGE_REFERENCE_MASK, IMAGE_DL_3D = load_data()
    IMAGE_CORRUPTED = IMAGE_REFERENCE * IMAGE_REFERENCE_MASK
    
    admm_adam_framework = ADMMADAM(IMAGE_REFERENCE, IMAGE_REFERENCE_MASK, IMAGE_CORRUPTED, IMAGE_DL_3D)
    image_recovery = admm_adam_framework.restore_image()
    
    # save as mat file, and use Matlab's function (src/plot_result/main.m) to show image
    savemat("src/data/image_recovery.mat", dict(image_recovery=image_recovery))