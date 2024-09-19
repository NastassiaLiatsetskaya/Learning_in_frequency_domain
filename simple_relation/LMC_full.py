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
kernels=[kernel1, kernel2, kernel3, kernel4, kernel5, kernel6 ]
kernel=LinearModelOfCoregionalizationKernel(kernels,output_dims= 22,input_dims=22,Q=6)



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
data1=[data10, data12]

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

    return [data10,data12]



data2=extract_points(y_train_real,2,3,x_train_real)
data3=extract_points(y_train_real,4,5,x_train_real)
data4=extract_points(y_train_real,6,7,x_train_real)
data5=extract_points(y_train_real,8,9,x_train_real)
data6=extract_points(y_train_real,10,11,x_train_real)
data7=extract_points(y_train_real,12,13,x_train_real)
data8=extract_points(y_train_real,14,15,x_train_real)
data9=extract_points(y_train_real,16,17,x_train_real)
data10=extract_points(y_train_real,18,19,x_train_real)
data11=extract_points(y_train_real,20,21,x_train_real)
data=mogptk.DataSet(data1+data2+data3+data4+data5+data6+data7+data8+data9+data10+data11)
print("created data set")
model=mogptk.Model(data,kernel )



model.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model.log_marginal_likelihood(),"trained")



model.save("LMC_20_full")
