import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import LinearKernel, ExponentialKernel,RationalQuadraticKernel, MaternKernel, PolynomialKernel
from mogptk.gpr.multioutput import LinearModelOfCoregionalizationKernel

torch.manual_seed(1)
import json


def func1(t):
    return torch.tensor([np.sin(t) + np.cos(t ** 2) + max(30, t)])


def func2(t):
    return torch.tensor((torch.tanh(((func1(t)) / 100) * 2) ))



x=np.linspace(1,500, 6000)
points1=[func1(i).numpy()[0] for i in x ]
points2=[func2(i).numpy()[0] for i in x]
conv_points=np.log(np.convolve(np.asarray(points1),np.asarray(points1), "same"))
points2=conv_points
f2=[]
f1=[]
i=0
while i< len(conv_points):
    f2.append(conv_points[i:i+6])
    f1.append(points1[i:i+6])
    i+=6


f1_real = []
f1_imag = []
f2_real = []
f2_imag = []
for i in range(1000):
    value1 = rfft(f1[i])
    value2 = rfft(f2[i])
    real_train=[]
    for i in value1:

        real_train.append(i.real)
        real_train.append(i.imag)
    f1_real.append(real_train)
    real_train2=[]
    for i in value2:
        real_train2.append(i.real)
        real_train2.append(i.imag)
    f2_real.append(real_train2)



x_train_real = torch.tensor(np.asarray(f1_real))




y_train_real = torch.tensor(np.asarray(f2_real))
print(y_train_real.shape, "shape")

kernel1=ExponentialKernel()
kernel2=MaternKernel()
kernel3=PolynomialKernel(2)
kernel4=RationalQuadraticKernel()
kernel5=ExponentialKernel()
kernel6=PolynomialKernel(3)
kernels=[kernel1, kernel2, kernel3, kernel4, kernel5, kernel6 ]
kernel=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=8,Q=6)
kernel_sec=LinearModelOfCoregionalizationKernel(kernels, output_dims=2,input_dims=8,Q=6)
kernel_third=LinearModelOfCoregionalizationKernel(kernels, output_dims= 2,input_dims=8,Q=6)
kernel_fourth=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=8,Q=6)

print(x_train_real.shape, "x train")
print(y_train_real[:2,0], "y train")
points_input=[]
for i in y_train_real[:,0]:
    points_input.append(i)

points_input2=[]
for i in y_train_real[:,1]:
    points_input2.append(i)
points_input=torch.tensor(np.asarray(points_input))
points_input2=torch.tensor(np.asarray(points_input2))
data10=mogptk.Data(x_train_real,points_input)
data12=mogptk.Data(x_train_real,points_input2)
data1=mogptk.DataSet([data10, data12])

def extract_points(points,i,j, x_train_real):
    points_input=[]
    for elem in points[:, i]:
        points_input.append(elem)
    points_input = torch.tensor(np.asarray(points_input))

    points_input2 = []
    for elem in points[:, j]:
        points_input2.append(elem)
    points_input2 = torch.tensor(np.asarray(points_input2))
    data10 = mogptk.Data(x_train_real, points_input)
    data12 = mogptk.Data(x_train_real, points_input2)
    data1 = mogptk.DataSet([data10, data12])
    return data1


model1=mogptk.Model(data1,kernel )
data2=extract_points(y_train_real,2,3,x_train_real)
model2=mogptk.Model(data2,kernel_sec)
data3=extract_points(y_train_real,4,5,x_train_real)
model3=mogptk.Model(data3, kernel_third)
data4=extract_points(y_train_real,6,7,x_train_real)
model4=mogptk.Model(data4, kernel_fourth)





model1.train(method="Adam", plot=True, iters=1000,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")
model2.train(method="Adam", plot=True, iters=1000,  error="MAE")
print(model2.log_marginal_likelihood(),"trained")
model3.train(method="Adam", plot=True, iters=1000,  error="MAE")
print(model3.log_marginal_likelihood(),"trained")
model4.train(method="Adam", plot=True, iters=1000, error="MAE")
print(model4.log_marginal_likelihood(),"trained")





model1.save("LMC1_2")
model2.save("LMC2_2")
model3.save("LMC3_2")
model4.save("LMC4_2")