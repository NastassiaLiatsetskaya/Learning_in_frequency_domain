import mogptk
from scipy.fft import rfft
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import SpectralMixtureKernel

torch.manual_seed(1)



def low_fidelity(x):
    return np.sin(8*np.pi*x)

def high_fidelity(x):
    return np.sin(8*np.pi*x)**2


x=np.linspace(0,1,480)
points1=[low_fidelity(i) for i in x ]
points2=[high_fidelity(i) for i in x]


f1_real=[]
for i, k in zip(x, points1):
    f1_real.append([i,k])
x_train_real = torch.tensor(np.asarray(f1_real))





y_train_real = torch.tensor(np.asarray(points2))

method="BNSE"



kernel1_r=SpectralMixtureKernel(Q=15,input_dims=2)




data1=mogptk.DataSet(x_train_real,y_train_real)
print(data1.get_input_dims(),"input dimensions data1")
model1=mogptk.SM(data1,Q=15)
model1.init_parameters(method)






model1.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")



model1.save("m1_r_sin_sm_bnse_one_point")