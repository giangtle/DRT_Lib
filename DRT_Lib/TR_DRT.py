import numpy as np
from math import pi
from math import log
from scipy.optimize import minimize
from scipy.optimize import Bounds
from .submodule import calc_A_re, calc_A_im
from .submodule import display_result


def TR_DRT(freq_vec, Z_exp, x0=None, el=1e-4, method="SLSQP", display=False):
    gamma, R_inf, loss = Tikhonov_minimization(freq_vec, Z_exp, x0, el, method)
    if display is True:
        display_result(freq_vec, Z_exp, gamma, R_inf)
    return gamma, R_inf, loss

def Tikhonov_minimization(freq_vec, Z_exp, x0, el, method):
    Z_exp_re=np.real(Z_exp)
    Z_exp_im=np.imag(Z_exp)
    A_re = calc_A_re(freq_vec)
    A_im = calc_A_im(freq_vec)
    if x0 is None:
        x0 = initial_guess(freq_vec, Z_exp)
    bounds = Bounds(np.zeros(x0.shape), np.abs(Z_exp).max()*np.ones(x0.shape))
    result = minimize(S, x0, args=(Z_exp_re, Z_exp_im, A_re, A_im, el), method=method,
                      bounds = bounds, options={'disp': True, 'ftol':1e-10, 'maxiter':200})
    gamma_R_inf = result.x
    MSE_re = np.sum((gamma_R_inf[-1] + np.matmul(A_re, gamma_R_inf[:-1]) - Z_exp_re)**2)
    MSE_im = np.sum((np.matmul(A_im, gamma_R_inf[:-1]) - Z_exp_im)**2)
    loss = MSE_re + MSE_im
    R_inf = gamma_R_inf[-1]
    gamma = gamma_R_inf[:-1]
    return gamma, R_inf, loss
    
def S(gamma_R_inf, Z_exp_re, Z_exp_im, A_re, A_im, el):
    MSE_re = np.sum((gamma_R_inf[-1] + np.matmul(A_re, gamma_R_inf[:-1]) - Z_exp_re)**2)
    MSE_im = np.sum((np.matmul(A_im, gamma_R_inf[:-1]) - Z_exp_im)**2)
    reg_term = el/2*np.sum(gamma_R_inf[:-1]**2)
    obj = MSE_re + MSE_im + reg_term
    return obj

def initial_guess(freq_vec, Z_exp):
    x0 = np.zeros(len(freq_vec)+1)
    x0[-1] = np.abs(Z_exp)[-1]
    return x0
