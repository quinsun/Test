""" Numerov method for one-dimensional box of length l well: psi''=-8*pi2*m*E/h2*psi
    reduce Er=E/(h2/m/l2), xr=x/l, psi_r=psi/l**0.5
    psi''=-8*pi2*Er*psi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def g(x,Er):
    f = np.zeros(len(x))
    f = -8*np.pi**2*Er*(x>-1)
    return f

def psi(x,g):
    s = x[1]-x[0]
    f = np.zeros(len(x))
    f[0], f[1] = 0, 0.01
    for i in range(2,len(x)):
        f[i] = (2*f[i-1]-f[i-2]+5/6*g[i-1]*f[i-1]*s*s+s*s/12*g[i-2]*f[i-2])/(1-s*s/12*g[i])      
    return f

def target(Er):
    return psi(x,g(x,Er))[-1]

def opt(x,Er):
    return least_squares(target, [Er])
    

if __name__ == "__main__":
    Er = 0.4    # h2/(m*L2)
    x = np.linspace(0,1,101)
    y = psi(x, g(x,Er))

    plt.plot(x,y)

    best = opt(x,Er)
    y2 = psi(x, g(x,best.x))
    plt.plot(x,y2, label = str(best.x))
    plt.legend()
    plt.show()
    
    
