import numpy as np
from .submodule import calc_A_re, calc_A_im


def calculate_EIS(freq_vec, gamma, R_inf):
    A_re = calc_A_re(freq_vec)
    A_im = calc_A_im(freq_vec)
    Z_cal = R_inf + np.matmul(A_re, gamma) + 1j*np.matmul(A_im, gamma)
    return Z_cal
