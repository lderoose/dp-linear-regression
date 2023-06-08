#!/usr/bin/python3.8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn



class CustomMSE(nn.Module):
    
    """
    Implementation of the objective function f<bar>D(w) in Algorithm 1.
    The default behavior is MSE. Set 'inject_noise=True' to add noise to lambda coefficients.
    """
    
    def __init__(self, x, y, **kwargs):
        super(CustomMSE, self).__init__()
        self.inject_noise = kwargs.get("inject_noise", False)
        
        self.lambda_psi0 = torch.sum(y**2)
        self.lambda_psi1 = -2*torch.matmul(y, x)
        self.lambda_psi2 = torch.sum(torch.matmul(x.unsqueeze(2), x.unsqueeze(1)), dim=0)
        
        if self.inject_noise:
            self.epsilon = kwargs.get("epsilon", 1)
            
            self.d = x.shape[1]
            self.delta = 2*(self.d+1)**2
            self.mu = 0  # 3. PRELIMINARIES
            self.scale = self.delta / self.epsilon
            laplace = torch.distributions.laplace.Laplace(self.mu, 1/self.scale)  # not self.scale 
        
            self.lambda_psi1 += laplace.sample((self.d,))
            self.lambda_psi2 += laplace.sample((self.d, self.d))

    def forward(self, w):       
        # phi0 = 1
        phi1 = w.flatten()
        phi2 = torch.outer(phi1, phi1)
        
        p0 = self.lambda_psi0 # * phi0
        p1 = torch.sum(torch.mul(self.lambda_psi1, phi1))
        p2 = torch.sum(torch.mul(self.lambda_psi2, phi2))

        loss = p0 + p1 + p2
        return loss
    
    
if __name__ == "__main__":
    # settings
    EPSILON = 1
    PATH_FIG = "./figure2"
    
    # data provided from example 4.2 Application in Linear Regression
    df_test = pd.DataFrame([[1, 0.4], [0.9, 0.3], [-0.5, -1]], columns=["x", "y"])  

    X = torch.from_numpy(df_test[["x"]].values)
    y = torch.from_numpy(df_test["y"].values) 
    loss = CustomMSE(X, y)
    a = loss.lambda_psi2.item()
    b = loss.lambda_psi1.item()
    c = loss.lambda_psi0.item()
    if (a == 2.06) & (b == -2.34) & (c == 1.25):
        print(f"OK <without noise>: fD(w) = {a}w² + ({b}w) + {c}")

    loss = CustomMSE(X, y, inject_noise=True, epsilon=EPSILON)
    a_prim = loss.lambda_psi2.item()
    b_prim = loss.lambda_psi1.item()
    c_prim = c  # no noise here
    if (a_prim != 2.06) & (b_prim != -2.34):
        print(f"OK <with noise injection>: fbarD(w) = {round(a_prim,2)}w² + ({round(b_prim,2)}w) + {c_prim}")
        
    # display figure 2
    ls_w = [e/100 for e in range(0, 101, 10)]
    ls_f = list(map(lambda w: a*w**2 + b*w + c, ls_w))
    ls_fbar = list(map(lambda w: a_prim*w**2 + b_prim*w + c_prim, ls_w))
    argmin_f = np.argmin(ls_f)
    argmin_fbar = np.argmin(ls_fbar)

    pl_f = plt.plot(ls_f, label=f"fD(w) = {a}w² + ({b}w) + {c}", c="blue")
    pl_fbar = plt.plot(ls_fbar, label=f"fbarD(w) = {round(a_prim,2)}w² + ({round(b_prim,2)}w) + {c_prim}", c="green", linestyle="dotted")
    plt.scatter(x=argmin_f, y=ls_f[argmin_f], c="blue", label="argmin f")
    plt.scatter(x=argmin_fbar, y=ls_fbar[argmin_fbar], c="green", label="argmin fbar")
    plt.xticks(np.arange(len(ls_w)), ls_w)
    plt.ylim((0.2, 1.6))
    plt.xlabel("w")
    plt.title("reproduction of Figure 2")
    plt.legend()
    plt.savefig(PATH_FIG);
