import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
from mogptk.gpr.multioutput import GaussianConvolutionProcessKernel
torch.manual_seed(1)
import json
def func1(t):
    return torch.tensor([np.sin(8*np.pi*t)])

def func3(t):
    return torch.tensor([np.sin(8*np.pi*t+np.pi/10)**2 + np.cos(4*np.pi*t)])


num_training_samples=400
length=6

x_train=[]
for i in range(num_training_samples):
    index=i+i*6
    x_train.append(np.linspace(index/2799, (index+6)/2799, 6))

f1=[]
f2=[]
for i in x_train:
    values1=[]
    values2=[]
    for index in i:
        values1.append(func1(index).item())
        values2.append(func3(index).item())
    f1.append(values1)
    f2.append(values2)
f1_real=[]
f1_imag=[]
f2_real=[]
f2_imag=[]

for i in range(num_training_samples):
    value1=rfft(f1[i])
    value2=rfft(f2[i])
    f1_real.append([i.real for i in value1])
    f1_imag.append([i.imag for i in value1])

    f2_real.append([i.real for i in value2])
    f2_imag.append([i.imag for i in value2])





x_train_real=torch.tensor(np.asarray(f1_real))

f1_imag_cleaned=np.asarray(f1_imag)#[:,1:3]
x_train_imag=torch.tensor(f1_imag_cleaned )

y_train_real=torch.tensor(np.asarray(f2_real))
y_train_imag=torch.tensor(np.asarray(f2_imag))


data_real1=mogptk.Data(x_train_real,y_train_real[:,0])
data_real2=mogptk.Data(x_train_real,y_train_real[:,1])
data_real3=mogptk.Data(x_train_real,y_train_real[:,2])
data_real4=mogptk.Data(x_train_real,y_train_real[:,3])
data_real=[data_real1,data_real2,data_real3,data_real4]


data_im1=mogptk.Data(x_train_imag,y_train_imag[:,0])
data_im2=mogptk.Data(x_train_imag,y_train_imag[:,1])
data_im3=mogptk.Data(x_train_imag,y_train_imag[:,2])
data_im4=mogptk.Data(x_train_imag,y_train_imag[:,3])
data_im=[data_im1,data_im2,data_im3,data_im4]

data_real=mogptk.DataSet(data_real)
data_imag=mogptk.DataSet(data_im)
print(data_real.get_input_dims())

model_real=mogptk.CONV(data_real,4)
model_imag=mogptk.CONV(data_imag,4)
print("start training")
model_real.train(method="Adam", iters=500, error="MAE")
print("first trained", model_real.log_marginal_likelihood())
model_imag.train(method="Adam", iters=500, error="MAE")
print("second trained", model_imag.log_marginal_likelihood())
model_real.save("model_real_convolved2")
model_imag.save("model_imag_convolved2")