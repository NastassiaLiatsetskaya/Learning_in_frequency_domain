import mogptk
from scipy.fft import rfft
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import SpectralMixtureKernel, ExponentialKernel, MaternKernel,PolynomialKernel,RationalQuadraticKernel, SquaredExponentialKernel
torch.manual_seed(1)


def high_fidelity(x):
    low_freq= np.sin(np.pi*x)+2*np.sin(2*np.pi*x)+ np.sin(3*np.pi*x)
    high_freq=np.sin(7*np.pi*x)+2*np.sin(8*np.pi*x)+ np.sin(9*np.pi*x)
    return low_freq+high_freq


x=np.linspace(0,10,800)
#points1=[low_fidelity(i) for i in x ]
points2=[high_fidelity(i) for i in x]


x_train_real = torch.tensor(x)





y_train_real = torch.tensor(np.asarray(points2))

method="BNSE"


kernel1_r=AddKernel([ExponentialKernel(input_dims=18), MaternKernel(input_dims=18),PolynomialKernel(2,input_dims=18), RationalQuadraticKernel(input_dims=18), SquaredExponentialKernel(input_dims=18)])




data1=mogptk.DataSet(x_train_real,y_train_real)
model1=mogptk.Model(data1, kernel1_r)





model1.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")


model1.save("m1_r_sin_orig_one_point")