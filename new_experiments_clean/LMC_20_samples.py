import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import LinearKernel, ExponentialKernel,RationalQuadraticKernel, MaternKernel, PolynomialKernel, ConstantKernel
from mogptk.gpr.multioutput import LinearModelOfCoregionalizationKernel

torch.manual_seed(1)
import json


def func1(t):
    return torch.tensor([np.sin(t) + np.cos(t ** 2) + max(30, t)])


def func2(t):
    return torch.tensor((torch.tanh(((func1(t)) / 100) * 2) ))




x=np.concatenate([np.linspace(100,200, 3340),
                  np.linspace(300,400, 3340),
                  np.linspace(400,500, 3340)])

points1=[func1(i).numpy()[0] for i in x ]
points2=[func2(i).numpy()[0] for i in x]
conv_points=np.log(np.convolve(np.asarray(points1),np.asarray(points1), "same"))
points2=conv_points
f2=[]
f1=[]
i=0
while i< len(conv_points):
    f2.append(conv_points[i:i+20])
    f1.append(points1[i:i+20])
    i+=20


f1_real = []
f1_imag = []
f2_real = []
f2_imag = []
for i in range(501):
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
print(x_train_real.shape, "x train shape")



y_train_real = torch.tensor(np.asarray(f2_real))
print(y_train_real.shape, "shape")

kernel1=ExponentialKernel()
kernel2=MaternKernel()
kernel3=PolynomialKernel(2)
kernel4=RationalQuadraticKernel()
kernel5=ExponentialKernel()
kernel6=PolynomialKernel(3)
print(kernel6.input_dims, "kernel 6")
kernel7=ConstantKernel()
kernels=[kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7 ]
kernel=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)
kernel_sec=LinearModelOfCoregionalizationKernel(kernels, output_dims=2,input_dims=22,Q=7)
kernel_third=LinearModelOfCoregionalizationKernel(kernels, output_dims= 2,input_dims=22,Q=7)
kernel_fourth=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)
kernel_fifth=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)
kernel_sixth=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)
kernel_seventh=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)
kernel_8=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)
kernel_9=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)
kernel_10=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)
kernel_11=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=22,Q=7)


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

data5=extract_points(y_train_real,8,9,x_train_real)
model5=mogptk.Model(data5, kernel_fifth)

data6=extract_points(y_train_real,10,11,x_train_real)
model6=mogptk.Model(data6, kernel_sixth)

data7=extract_points(y_train_real,12,13,x_train_real)
model7=mogptk.Model(data7, kernel_seventh)

data8=extract_points(y_train_real,14,15,x_train_real)
model8=mogptk.Model(data8, kernel_8)

data9=extract_points(y_train_real,16,17,x_train_real)
model9=mogptk.Model(data9, kernel_9)

data10=extract_points(y_train_real,18,19,x_train_real)
model10=mogptk.Model(data10, kernel_10)

data11=extract_points(y_train_real,20,21,x_train_real)
model11=mogptk.Model(data11, kernel_11)

model1.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")
model2.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model2.log_marginal_likelihood(),"trained")
model3.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model3.log_marginal_likelihood(),"trained")
model4.train(method="Adam", plot=True, iters=500, error="MAE")
print(model4.log_marginal_likelihood(),"trained")
model5.train(method="Adam", plot=True, iters=500, error="MAE")
print(model5.log_marginal_likelihood(),"trained")
model6.train(method="Adam", plot=True, iters=500, error="MAE")
print(model6.log_marginal_likelihood(),"trained")
model7.train(method="Adam", plot=True, iters=500, error="MAE")
print(model7.log_marginal_likelihood(),"trained")
model8.train(method="Adam", plot=True, iters=500, error="MAE")
print(model8.log_marginal_likelihood(),"trained")
model9.train(method="Adam", plot=True, iters=500, error="MAE")
print(model9.log_marginal_likelihood(),"trained")
model10.train(method="Adam", plot=True, iters=500, error="MAE")
print(model10.log_marginal_likelihood(),"trained")
model11.train(method="Adam", plot=True, iters=500, error="MAE")
print(model11.log_marginal_likelihood(),"trained")


model1.save("LMC1_20_const")
model2.save("LMC2_20_const")
model3.save("LMC3_20_const")
model4.save("LMC4_20_const")
model5.save("LMC5_20_const")
model6.save("LMC6_20_const")
model7.save("LMC7_20_const")
model8.save("LMC8_20_const")
model9.save("LMC9_20_const")
model10.save("LMC10_20_const")
model11.save("LMC11_20_const")
