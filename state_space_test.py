import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci_int

def lorenz_deriv(vec,t0,sigma=10., beta=8./3, rho=28.0):
    return [sigma * (vec[1] - vec[0]), vec[0] * (rho - vec[2]) - vec[1], vec[0] * vec[1] - beta * vec[2]]

#call it a time series in 1-space
#bam, taken's embedding
#check rightness of embedding

#then, we can calculate M(x|y), which is not a trivial task

if __name__ == "__main__":
    x0 = [-8,-8,27]
    times = np.linspace(0,100,2**13)
    lorenz_gen = sci_int.odeint(lorenz_deriv, x0, times)
    lorenz_gen = lorenz_gen[3000:] #burn in
    plt.plot(lorenz_gen[:,0])
    plt.show()
