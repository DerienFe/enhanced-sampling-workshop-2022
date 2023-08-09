import numpy as np
import matplotlib.pyplot as plt

barrier_height = 0.7
gamma = 0.01
def initial_2D():
    N = 13
    x,y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
    Z = 5 - barrier_height * np.log((np.exp(-(x+2)**2 - 3*(y+2)**2))/gamma + 
                                    (np.exp(-5*(x-2)**2 - (y-2)**2))/gamma +
                                    (np.exp(-6*(x-2)**2 - 5*(y+2)**2))/gamma+
                                    (np.exp(-3*(x+2)**2 - (y-2)**2))/gamma 
                                    )
    #zero the Z
    Z = Z - np.min(Z)
    #plot Z
    plt.figure()
    plt.contourf(x,y,Z)
    plt.colorbar()
    plt.show()
    return

initial_2D()
print("hello")
                            