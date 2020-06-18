import numpy as np
import math
from math import pi
from math import log
import torch
import torch.nn.functional as F
from .submodule import calc_A_re, calc_A_im
from .submodule import display_result


def DP_DRT(freq_vec, Z_exp, lambda_limit=1e-4, learning_rate=1e-5, display=False):
    gamma, R_inf, loss = train_model(freq_vec, Z_exp, lambda_limit, learning_rate)
    if display is True:
        display_result(freq_vec, Z_exp, gamma, R_inf)
    return gamma, R_inf, loss

def train_model(freq_vec, Z_exp, lambda_limit, learning_rate):
    N_freqs = len(freq_vec)
    A_re = calc_A_re(freq_vec)
    A_im = calc_A_im(freq_vec)
    # transform impedance variables & DRT matrices into tensors
    Z_exp_re_torch = torch.from_numpy(np.real(Z_exp)).type(torch.FloatTensor).reshape(1,N_freqs)
    Z_exp_im_torch = torch.from_numpy(np.imag(Z_exp)).type(torch.FloatTensor).reshape(1,N_freqs)
    A_re_torch = torch.from_numpy(A_re.T).type(torch.FloatTensor)
    A_im_torch = torch.from_numpy(A_im.T).type(torch.FloatTensor)
    # create Deep Prior model
    vanilla_model = make_dp_model(N_freqs)
    model = vanilla_model()
    # model variables: random constant for input node (zeta), learning rate, optimizer
    zeta = torch.randn(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Regularization/stopping criteria: 
    # 1.lambd = |(new_loss-old_loss)/old_loss| < 1e-4
    # 2. max_iteration = 100,001
    iteration=0
    lambd=1
    old_loss=1
    while iteration < 100001 and lambd > lambda_limit:
        # Forward pass: compute predicted y by passing x to the model.
        gamma_torch = model(zeta)
        # Compute the loss
        loss = loss_fn(gamma_torch, Z_exp_re_torch, Z_exp_im_torch, A_re_torch, A_im_torch)
        # zero all gradients (purge any cache)
        optimizer.zero_grad()
        # compute the gradient of the loss with respect to model parameters
        loss.backward()
        # Update the optimizer
        optimizer.step()
        # Stop conditions:
        iteration = iteration +1
        if iteration > 5000:
            lambd = abs((loss.item()-old_loss)/old_loss)
        old_loss = loss.item()
    if iteration >= 100000:
        print("Max iteration reached")
    else:
        print("Early stop. Number of iteration: ",str(iteration))
    gamma = gamma_torch.detach().numpy().reshape(-1)
    R_inf = gamma[-1]
    gamma = gamma[:-1]
    return gamma, R_inf, old_loss
    
def make_dp_model(N_freqs):
    """
    Create the base deep prior model, "vanilla model", returns as a class object 
    """
    D_in = 1
    D_out = N_freqs+1
    class vanilla_model(torch.nn.Module):
        def __init__(self):
            super(vanilla_model, self).__init__()
            self.fct_1 = torch.nn.Linear(D_in, N_freqs)
            self.fct_2 = torch.nn.Linear(N_freqs, N_freqs)
            self.fct_3 = torch.nn.Linear(N_freqs, N_freqs)
            self.fct_4 = torch.nn.Linear(N_freqs, D_out)
            # initialize the weight parameters
            torch.nn.init.zeros_(self.fct_1.weight)
            torch.nn.init.zeros_(self.fct_2.weight)
            torch.nn.init.zeros_(self.fct_3.weight)
            torch.nn.init.zeros_(self.fct_4.weight)
        # forward
        def forward(self, zeta):
            h = F.elu(self.fct_1(zeta))
            h = F.elu(self.fct_2(h))
            h = F.elu(self.fct_3(h))
            gamma_pred = F.softplus(self.fct_4(h), beta = 5)
            return gamma_pred
    return vanilla_model
    
def loss_fn(output, Z_exp_re_torch, Z_exp_im_torch, A_re_torch, A_im_torch):
    """ 
    Loss function of the DRT fit
    """
    MSE_re = torch.sum((output[:, -1] + torch.mm(output[:, 0:-1], A_re_torch) - Z_exp_re_torch)**2)
    MSE_im = torch.sum((torch.mm(output[:, 0:-1], A_im_torch) - Z_exp_im_torch)**2)
    MSE = MSE_re + MSE_im
    return MSE