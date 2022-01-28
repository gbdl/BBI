import sys
sys.path.append("..")

import numpy as np
import torch
from matplotlib import pyplot as plt
from inflation import  BBI

# some plotting functions
# This is the distance from the origin
def distance(xs): return torch.norm(xs)

# plots a given function
def plotting(f, xslist, name):
    to_plot = []

    for point in xslist:
        to_plot.append(f(torch.tensor(point)))
        
    to_plot = torch.stack(to_plot)
    with torch.no_grad():
        plt.title(name, fontsize=15)
        plt.plot(to_plot)
        plt.grid()
        plt.show()

def plotting2(f, xslist,xslist2, name):
    to_plot = []
    to_plot2 = []
    
    for point in xslist:
        to_plot.append(f(torch.tensor(point)))
    for point in xslist2:
        to_plot2.append(f(torch.tensor(point)))
        
    to_plot = torch.stack(to_plot)
    with torch.no_grad():
        plt.title(name, fontsize=15)
        plt.plot(to_plot, '--')
        plt.plot(to_plot2)
        plt.grid()
        plt.show()   
        
    
## BBI
def BBI_optimizer(x0, func, iterations = 1000, lr=0.0001, threshold0=50, threshold=1000000, v0=1e-10, deltaEn =.0, n_fixed_bounces = 1, consEn = True):
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = BBI([xs], lr=lr, threshold0 = int(threshold0), threshold = int(threshold), v0 = v0, deltaEn = deltaEn, consEn = consEn, n_fixed_bounces = n_fixed_bounces)
    xs_best = torch.tensor(x0)
    minloss = np.inf
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn=func(xs)
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
            xs_best = xs.detach().clone()
        loss_fn.backward()
        def closure():
                    return loss_fn
        optimizer.step(closure)
    return xs_best

def BBI_optimizer_fullhistory(x0, func, iterations = 1000, lr=0.0001, threshold0=50, threshold=1000000, v0=1e-10, deltaEn =.0, n_fixed_bounces = 1, consEn = True):
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = BBI([xs], lr=lr, threshold0 = int(threshold0), threshold = int(threshold), v0 = v0, deltaEn = deltaEn, consEn = consEn, n_fixed_bounces = n_fixed_bounces)
    minloss = np.inf
    xslist = []
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn=func(xs)
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
        loss_fn.backward()
        def closure():
                    return loss_fn
        optimizer.step(closure)
        xslist.append(xs.tolist())
    return xslist


## sgd
def sgd_optimizer(x0, func, iterations = 1000, lr=0.0001, momentum=.95,):
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = torch.optim.SGD([xs], lr=lr, momentum = momentum)
    xs_best = torch.tensor(x0)
    minloss = np.inf
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn=func(xs)
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
            xs_best = xs.detach().clone()
        loss_fn.backward()
        optimizer.step()
    return xs_best

def sgd_optimizer_fullhistory(x0, func, iterations = 1000, lr=0.0001, momentum=.95):
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = torch.optim.SGD([xs], lr=lr, momentum = momentum)
    minloss = np.inf
    xslist = []
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn=func(xs)
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
        loss_fn.backward()
        optimizer.step()
        xslist.append(xs.tolist())
    return xslist

#sgd-gamma: used for tuning -log(momentum)

def sgd_optimizer_gamma(x0, func, iterations = 1000, lr=0.0001, gamma=100):
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = torch.optim.SGD([xs], lr=lr, momentum = np.exp(-gamma))
    xs_best = torch.tensor(x0)
    minloss = np.inf
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn=func(xs)
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
            xs_best = xs.detach().clone()
        loss_fn.backward()
        optimizer.step()
    return xs_best


def sgd_optimizer_gamma_fullhistory(x0, func, iterations = 1000, lr=0.0001, gamma=100):
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = torch.optim.SGD([xs], lr=lr, momentum = np.exp(-gamma))
    minloss = np.inf
    xslist = []
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn=func(xs)
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
        loss_fn.backward()
        optimizer.step()
        xslist.append(xs.tolist())
    return xslist