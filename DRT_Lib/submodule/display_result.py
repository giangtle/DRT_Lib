import numpy as np
import matplotlib.pyplot as plt
from math import pi, log


def display_result(freq_vec, Z_exp, gamma, R_inf, sigma_gamma = None):
    Z_cal = calculate_EIS(freq_vec, gamma, R_inf)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1.plot(np.real(Z_exp), -np.imag(Z_exp), linewidth=2, color="red", label="$Z_{exp}$")
    ax1.plot(np.real(Z_cal), -np.imag(Z_cal), "ko", label="$Z_{cal}$")
    ax1.legend(frameon=False, fontsize = 15)
    ax1.set_xlabel(r'$Real[Z_{11}]$', fontsize = 15)
    ax1.set_ylabel(r'$-Imag[Z_{11}]$', fontsize = 15)
    if sigma_gamma is not None:
        plt.fill_between(freq_vec, gamma-3*sigma_gamma, gamma+3*sigma_gamma,
                         color="0.4", alpha=0.3, label="confidence interval")
    ax1.axis("equal")
    ax2.semilogx(freq_vec, gamma, color='red', marker='o', markeredgecolor="black",
                 markerfacecolor='black', linewidth=2, label="$DRT$")
    ax2.legend(frameon=False, fontsize = 15)
    ax2.set_xlabel(r'$f / Hz$', fontsize = 15)
    ax2.set_ylabel(r'$\gamma/\Omega$', fontsize = 15)
    plt.show()
    return

def calculate_EIS(freq_vec, gamma, R_inf):
    A_re = calc_A_re(freq_vec)
    A_im = calc_A_im(freq_vec)
    Z_cal = R_inf + np.matmul(A_re, gamma) + 1j*np.matmul(A_im, gamma)
    return Z_cal

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

def calc_A_re(freq):
    omega = 2.*pi*freq
    tau = 1./freq
    N_freqs = freq.size
    out_A_re = np.zeros((N_freqs, N_freqs))
    for p in range(0, N_freqs):
        for q in range(0, N_freqs):
            if q == 0:
                out_A_re[p, q] = -0.5/(1+(omega[p]*tau[q])**2)*log(tau[q+1]/tau[q])
            elif q == N_freqs-1:
                out_A_re[p, q] = -0.5/(1+(omega[p]*tau[q])**2)*log(tau[q]/tau[q-1])
            else:
                out_A_re[p, q] = -0.5/(1+(omega[p]*tau[q])**2)*log(tau[q+1]/tau[q-1])
    return out_A_re