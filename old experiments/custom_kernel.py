import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
torch.manual_seed(1)
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import LinearKernel, ExponentialKernel, CosineKernel, WhiteKernel, MaternKernel
def func1(t):
    return torch.tensor([np.sin(t)+ np.cos(t**2)+ max(30,t)])
   # return t*2
def func2(t):
    return (torch.relu(func1(t))+ torch.sigmoid(func1(t)))
    #return t*3

num_training_samples=400
length=5

x_train=[]
for i in range(num_training_samples):
    index=i+i*5
    x_train.append(np.linspace(index, index+5, 5))


f1=[]
f2=[]
for i in x_train:
    values1=[]
    values2=[]
    for index in i:
        values1.append(func1(index).item())
        values2.append(func2(index).item())
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


x_test=[411,412,413,414,415]
values=[func1(x) for x in x_test]
values_second=[func2(x) for x in x_test]
print(values, "first values")
print(values_second, "second values")
print(rfft(values_second), "output")
x_train_real=torch.tensor(np.asarray(f1_real))
x_train_imag=torch.tensor(np.asarray(f1_imag))

y_train_real=torch.tensor(np.asarray(f2_real))

y_train_imag=torch.tensor(np.asarray(f2_imag))



kernel_real= AddKernel([ExponentialKernel(), CosineKernel(),MaternKernel(), LinearKernel()])
kernel_real2= AddKernel([ExponentialKernel(), CosineKernel(), MaternKernel()])
kernel_real3= AddKernel([ExponentialKernel(),CosineKernel(),MaternKernel()])



data1=mogptk.DataSet(x_train_real,y_train_real[:,0])
#model1=mogptk.SM(data1,Q=20)
data2=mogptk.DataSet(x_train_real,y_train_real[:,1])

data3=mogptk.DataSet(x_train_real,y_train_real[:,2])

model_s1=mogptk.Model(data1,kernel_real)
model_s1.train(method="Adam", iters=500, plot=True, error= "MAE")
model_s2=mogptk.Model(data2,kernel_real2)
model_s2.train(method="Adam",  iters=500, plot=True, error= "MAE")
model_s3=mogptk.Model(data3,kernel_real3)
model_s3.train(method="Adam",  iters=500, plot=True, error= "MAE")


x_test=torch.tensor([[x.real for x in rfft(values)]])
print(x_test.shape,"shaoe")


x, y, low_bound, upper_bound=model_s1.predict(x_test)
x1, y1, _, _=model_s2.predict(x_test)
x2, y2, low_bound2, upper_bound2=model_s3.predict(x_test)
print(y,"y first")
print(y1, "y second")
print(y2, "y third")
print(low_bound,"low_bound")
print(upper_bound,"upper_bound")