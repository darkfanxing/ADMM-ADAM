import numpy as np
from scipy.io import loadmat

def get_matlab_variable(mat_file_path):
    matlab_variable = loadmat(mat_file_path)

    matlab_variable_name = list(matlab_variable.keys())[-1]
    return matlab_variable[matlab_variable_name]

def load_data():
    return \
        get_matlab_variable("src/data/Ottawa.mat"), \
        get_matlab_variable("src/data/mask_Ottawa.mat"), \
        get_matlab_variable("src/data/X3DL.mat")