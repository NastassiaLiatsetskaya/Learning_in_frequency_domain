import mogptk
from scipy.fft import rfft
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import SpectralMixtureKernel

torch.manual_seed(1)


def func_test(x):
    low_freq= np.sin(np.pi*x)+2*np.sin(2*np.pi*x)+ np.sin(3*np.pi*x)
    high_freq=np.sin(7*np.pi*x)+2*np.sin(8*np.pi*x)
    return low_freq+high_freq



x=np.linspace(0,10,500)
points2=[func_test(i) for i in x]


x_train_real = torch.tensor(x)





y_train_real = torch.tensor(np.asarray(points2))

method="BNSE"


kernel1_r=SpectralMixtureKernel(Q=15,input_dims=1)



data1=mogptk.DataSet(x_train_real,y_train_real)
model1=mogptk.Model(data1,kernel1_r)





model1.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")


model1.save("m1_sin_sm_orig_one_point")