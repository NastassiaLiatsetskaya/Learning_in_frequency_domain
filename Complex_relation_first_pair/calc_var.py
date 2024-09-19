import numpy as np
def calculate_variance(n, real_var, imag_var, N):
    variance=(1/(N**2))*real_var[int(0)]+ (1/(N**2))*np.cos(n*np.pi)**2 *real_var[int(4)]
    k =1
    while k<=(int(4) -1):
        variance+=(4/N**2) *((np.cos(n*np.pi*2*k/N))**2 *real_var[k] + np.sin(n*np.pi*2*k/N)**2 *imag_var[k])
        k+=1
    return variance

def calculate_mean(n, real_mean, imag_mean, N):
    mean=(1/N)*real_mean[0]+ (1/N)*np.cos(n*np.pi)*real_mean[4]
    k=1
    while k<=(3):
        mean+=(2/N) *((np.cos(n*np.pi*2*k/N)) *real_mean[k] - np.sin(n*np.pi*2*k/N) *imag_mean[k])
        k+=1
    return mean