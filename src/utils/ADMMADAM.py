import numpy as np
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

    The ADMM-ADAM workflow
    ----------

    1. Train a Deep Learning model with ADAM optimizer (GAN in here),
       and get a Deep Learning olution (image)
    2. Extract the most important information of Deep Learning solution
       by PCA (or some other feature extractor)
    3. design a convex optimization problem with ADMM optimizer as follows:

            X^{*} = \displaystyle\arg\min_{X} ∥[X-Y]_{Ω}∥^{2}_{2} + λ/2 ∥X - X_{DL}∥^{2}_{Q}
    
        where, 
        - X:
            - X ∈ R^{M x L} be an M-band hyperspectral (target) image
              with L pixels
        - Y:
            - Y ∈ R^{M x L} be the observed image, meaning that some of
              its entries are missing
        - Ω:
            - Ω ⊆ {1, ..., ML} denote the index set of those available data.
        - X_{DL}:
            - X_{DL} ∈ R^{M x L} can be obtained using ADAM optimizer (GAN
              model with ADAM optimizer in here)
        - λ:
            - λ > 0 is called regularization parameter empirically set as 0.01
              in this work
        - Q:
            - Q-qudratic norm, which extracts useful features from x_{DL} for
              effective regularization
            - feature extractor is PCA in here

        Assume we have N materials, each pixel can be modeled as a linear
        combination of N spectral signature vectors in R^{M}. In other words,
        all the hyperspectral pixel vectors belong to a N-dimensional subspace
        if we ignore some non-linearity or noise effects, so the target image
        X can be represented as follows:
            
            X=ES
        
        where,
        - E: the most important N component (eigenvector), i.e. N material
        - S: some coefficient matrix S ∈ R^{N x L}

        and we can simplify the objective function:

            ∵
                - E is a semiunitary matrix, E^{T} E = I_N
                - ∥v∥^{2}_{Q} = v^{T}Qv
                - X = ES
            
            ∴
                ∥X - X_{DL}∥^{2}_{Q}
                = (X - X_{DL})^{T}(X - X_{DL})
                = X^{T} - 2*X_{DL} + X_{DL}^{T}X_{DL}
                = S^{T}E^{T}ES - 2*E_{DL} S_{DL} + S_{DL}^{T} E_{DL}^{T}E_{DL}S_{DL}
                = ∥S - S_{DL}∥^{2}_{F}
        
        so convex optimization problem can be represented as follows:

            S^{*} = \displaystyle\arg\min_{S} ∥[ES-Y]_{Ω}∥^{2}_{F} + λ/2 ∥S - S_{DL}∥^{2}_{F}
    
        - the meaning of F-norm is:
            ∥X∥^{2}_{F} = ∥vec(X)∥^{2}_{2}

        Once S^{⋆} is available, it can be used to reconstruct the
        complete hyperspectral image as X = ES^{*}

        Hehe... it is happy time for reformulating convex optimization
        problem into the standard ADMM form:

            \min_{Z=S} ∥[ES-Y]_{Ω}∥^{2}_{F} + λ/2 ∥Z - S_{DL}∥^{2}_{F}
        
        and give a augmented Lagrangian term (μ/2 ∥S-Z-D∥^2_{F}):
            \min_{Z=S} ∥[ES-Y]_{Ω}∥^{2}_{F} + λ/2 ∥Z - S_{DL}∥^{2}_{F} + μ/2 ∥S-Z-D∥^2_{F}
        
        where,
        - D:
            - D ∈ R^{N x L} is the scaled dual variable
        - μ
            - µ is the penalty parameter, empirically set as 0.001

        Then, ADMM optimizer solves the problem as detailed as follows:

        ----------------
        Given λ > 0, µ > 0, and xDL ≡ XDL. Initialize S^{0} := 0 (or by
        warm start), D^{0} := 0, Z^{0} := 0.

        repeat
            Z^{q+1} := (λ / (λ + µ)) * S_{DL} +  (µ/(λ + µ)) * (S_q - D_q)
            S^{q+1} := vec(S^{q+1}) = (RR^{T} + µ/2*I_{NL})^{-1}(RPy + µ/2*δ)
            D^{q+1} := = D_{q} - S_{q+1} + Z{q+1}
        until

        Output S^{*} = S^{q} 
        ----------------

        where,
        - R
            - R: ≜ (I_L ⊗ E^})P^T
        - δ
            - δ ≜ vec(Z_{q+1} + D_q), and P ∈ R^{|Ω| x (ML)
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