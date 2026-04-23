"""From-scratch kernels: RBF, Periodic, String."""
import numpy as np
from collections import Counter

def rbf(X, Y, length=1.0, var=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:,None] + np.sum(Y**2, 1)[None,:] - 2*X@Y.T
    return var * np.exp(-0.5 * d2 / length**2)

def periodic(X, Y, length=1.0, period=1.0, var=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d = np.abs(X[:,None,:] - Y[None,:,:]).sum(-1)
    return var * np.exp(-2 * np.sin(np.pi * d / period)**2 / length**2)

def string_kernel(A, B, n=2):
    """Simple n-gram set kernel over whitespace-tokenised strings."""
    def grams(s):
        toks = s.split()
        return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    gA = [grams(s) for s in A]
    gB = [grams(s) for s in B]
    K = np.zeros((len(A), len(B)))
    for i,ga in enumerate(gA):
        for j,gb in enumerate(gB):
            K[i,j] = sum(ga[g]*gb[g] for g in ga.keys() & gb.keys())
    # normalise
    dA = np.sqrt(np.diag(string_kernel_self(gA))); dB = np.sqrt(np.diag(string_kernel_self(gB)))
    return K / (dA[:,None]*dB[None,:] + 1e-12)

def string_kernel_self(g_list):
    K = np.zeros((len(g_list), len(g_list)))
    for i,ga in enumerate(g_list):
        for j,gb in enumerate(g_list):
            K[i,j] = sum(ga[g]*gb[g] for g in ga.keys() & gb.keys())
    return K
