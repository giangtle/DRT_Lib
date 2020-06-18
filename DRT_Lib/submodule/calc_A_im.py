from math import pi
from math import log
import numpy as np


def calc_A_im(freq):
    omega = 2.*pi*freq
    tau = 1./freq
    N_freqs = freq.size
    out_A_im = np.zeros((N_freqs, N_freqs))
    for p in range(0, N_freqs):
        for q in range(0, N_freqs):
            if q == 0:
                out_A_im[p, q] = 0.5*(omega[p]*tau[q])/(1+(omega[p]*tau[q])**2)*log(tau[q+1]/tau[q])
            elif q == N_freqs-1:
                out_A_im[p, q] = 0.5*(omega[p]*tau[q])/(1+(omega[p]*tau[q])**2)*log(tau[q]/tau[q-1])
            else:
                out_A_im[p, q] = 0.5*(omega[p]*tau[q])/(1+(omega[p]*tau[q])**2)*log(tau[q+1]/tau[q-1])
    return out_A_im