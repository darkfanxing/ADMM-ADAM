from numpy import ndarray, zeros, reshape, transpose, where, einsum, eye, dot
from numpy.linalg import inv, eigh
from scipy.sparse import csc_matrix, block_diag

class ADMMADAM():
    """
    ADMM-ADAM, A algorithm for image restoration by extract
    Deep Learning solution

    Reference
    ----------
    
    Lin, Chia-Hsiang, Yen-Cheng Lin, and Po-Wei Tang.
    "ADMM-ADAM: A New Inverse Imaging Framework Blending
    the Advantages of Convex Optimization and Deep Learning."
    IEEE Transactions on Geoscience and Remote Sensing (2021).

    Parameters
    ----------
    image_reference_3d: np.ndarray
        the ground truth image, it should be the same shape as:
            (image_width, image_height, image_band)

    image_reference_mask_3d: np.ndarray
        the mask of ground truth image, it should be the same shape as:
            (image_width, image_height, image_band)

    image_corrupted_3d: np.ndarray
        the image that miss data, it should be the same shape as:
            (image_width, image_height, image_band)

    image_dl_3d: np.ndarray
        the Deep Learning solution (image), it should be the same shape as:
            (image_width, image_height, image_band)

    Methods
    ---------
    restore_image() -> np.ndarray
        restore image by ADMM-ADAM framework, and then return the recovery
        image

    Examples
    ---------
    >>> from utils import load_data
    >>> IMAGE_REFERENCE, IMAGE_REFERENCE_MASK, IMAGE_DL_3D = load_data()
    >>> IMAGE_CORRUPTED = IMAGE_REFERENCE * IMAGE_REFERENCE_MASK
    >>> admm_adam_framework = ADMMADAM(
    ...     IMAGE_REFERENCE,
    ...     IMAGE_REFERENCE_MASK,
    ...     IMAGE_CORRUPTED,
    ...     IMAGE_DL_3D
    ... )
    >>> image_recovery = admm_adam_framework.restore_image(
    ...     n=10,
    ...     regularizer_lambda=0.01,
    ...     mu=1e-3,
    ...     iteration=50
    ... )
    """
    
    def __init__(self, image_reference_3d, image_reference_mask_3d, image_corrupted_3d, image_dl_3d):
        self.image_reference_3d: ndarray = image_reference_3d
        self.image_reference_mask_3d: ndarray = image_reference_mask_3d
        self.image_corrupted_3d: ndarray = image_corrupted_3d
        self.image_dl_3d: ndarray = image_dl_3d

        self._initial_image_shape()

        self.image_reference_mask_2d: ndarray = transpose(reshape(self.image_reference_mask_3d, (-1, self.image_band_number), "F"))
        self.image_corrupted_2d: ndarray = transpose(reshape(self.image_corrupted_3d, (-1, self.image_band_number), "F"))
        self.image_dl_2d: ndarray = transpose(reshape(self.image_dl_3d, (65536, self.image_band_number), order="F"))


    def restore_image(self, n=10, regularizer_lambda=0.01, mu=1e-3, iteration=50):
        """
        restore image by ADMM-ADAM framework, and then return the recovery
        image
        
        Parameters
        ---------
        n: int
            the number of the most important component of image you want

        regularizer_lambda: float
            the regularizer of convex optimization problem, empirically set as 0.01

        mu: float
            the penalty parameter of augmented Lagrangian in ADMM form

        iteration: int
            run optimization algorithm until reaches specific iteration
        """
        
        E = self._principal_components_analysis(n) # get n most important component

        S_DL = transpose(E).dot(self.image_dl_2d) # the meaning of "s_dl" is get most important n x_dl component by PCA, X=ES, S=E^{T}X

        # get RPy that the part of right block (RPy + μ/2*δ) of s
        P_TRANSPOSE_P = self._get_p_transpose_p() # P^T * P
        RP = einsum('kij, lk -> lij', P_TRANSPOSE_P, transpose(E))
        rpy = zeros((n, self.image_pixel_number))
        for pixel_index in range(self.image_pixel_number):
            rpy[:, pixel_index] = dot(RP[:, :, pixel_index], self.image_corrupted_2d[:, pixel_index])
        rpy = rpy.reshape((-1, 1), order="F") # (-1, 1)'s meaning is reshape to (655360, 1)
        
        # get the left block (RR^T + μ/2*I_{NL})^-1 of s
        s_left_block = self._get_s_left_block(RP, E, mu, n)

        s_old = zeros(S_DL.shape) # s_q in paper
        d_old = zeros(S_DL.shape) # d_q in paper
        for _ in range(iteration):
            # update to z_{q+1}
            z_new = \
                regularizer_lambda / (regularizer_lambda + mu) * S_DL \
                + mu / (regularizer_lambda + mu) * s_old - d_old

            # update to s_{q+1}, left block of s dot product right block (RPy + μ/2*δ) of s
            s_old = s_left_block.dot(rpy + mu/2 * reshape((z_new + d_old), (-1, 1), order="F"))
            s_old = reshape(s_old, (n, self.image_pixel_number), order="F")            
            
            # update to d_{q+1}
            d_old += -s_old + z_new

        # 1. S is optimized now, it can be converted to "X" by "X = ES"
        #     - the meaning of "X" is recovery image in here
        # 2. convert "X" to 3-D image after it converted to "X"
        return reshape(
            transpose(E.dot(reshape(s_old, (n, self.image_pixel_number), "F"))),
            self.image_reference_3d.shape,
            order="F"
        )

    def _get_s_left_block(self, rp, e, mu, n) -> ndarray:
        # 
        # RR^T
        #   = (I_L ⊗ E^T)P^{T}P(I_L ⊗ E^T)
        # 
        #       ∵ 
        #           formula: (B^T ⊗ A) vec(X) = vec(AXB)
        #           RP = (I_L ⊗ E^T)P^{T}P
        #       ∴ 
        #           (I_L ⊗ E^T) = (I_L ⊗ E^T) I_L = vec(E^{T}I_{L}I_{L}) = vec(E^T)
        # 
        #   = (I_L ⊗ E^T)P^{T}P(vec(E^T))
        #   = RP(vec(E^T))
        RR_TRANSPOSE = einsum('kij, li -> klj', rp, transpose(e))
        RR_TRANSPOSE_PERMUTED = transpose(RR_TRANSPOSE, (2, 0, 1)) # reshape to (65536, 10, 10)
        identity_matrix = (mu/2) * eye(n)

        s_left_block = zeros(RR_TRANSPOSE_PERMUTED.shape)
        for pixel_index in range(RR_TRANSPOSE_PERMUTED.shape[0]):
            s_left_block[pixel_index, :, :] = inv(RR_TRANSPOSE_PERMUTED[pixel_index, :, :] + identity_matrix)

        # Split Tensor to 65536 (10*10) matrix
        s_left_block = [csc_matrix(s_left_block[n, :, :]) for n in range(s_left_block.shape[0])]
        s_left_block = block_diag((s_left_block))
        return s_left_block

    def _get_p_transpose_p(self) -> ndarray:
        p_transpose_p: ndarray = zeros((self.image_band_number, self.image_band_number, self.image_pixel_number))
        available_pixel_x, available_pixel_y, available_band = where(self.image_reference_mask_3d==1)
        p_transpose_p[
            available_band,
            available_band,
            (available_pixel_x + self.image_row_length * available_pixel_y)
        ] = 1

        return p_transpose_p


    def _principal_components_analysis(self, n: int) -> ndarray:
        x = self.image_dl_3d.reshape((-1, self.image_band_number), order="F")
        _, eigenvector = eigh(transpose(x).dot(x))

        return eigenvector[:, self.image_band_number-n:]


    def _initial_image_shape(self) -> None:
        self.image_row_length, image_column_length, self.image_band_number = self.image_dl_3d.shape
        self.image_pixel_number: int = self.image_row_length * image_column_length
