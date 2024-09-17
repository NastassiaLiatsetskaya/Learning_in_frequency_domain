import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import SquaredExponentialKernel, ExponentialKernel,RationalQuadraticKernel, MaternKernel, PolynomialKernel,ConstantKernel
from mogptk.gpr.multioutput import LinearModelOfCoregionalizationKernel

torch.manual_seed(1)
import json


def func1(t):
    return torch.tensor([np.sin(t) + np.cos(t ** 2) + max(30, t)])


def func2(t):
    return torch.tensor((torch.tanh(((func1(t)) / 100) * 2) ))



def generate_points(start,stop,num_points,subsequence):
    eps = 0.1
    len = stop-start

    step = eps * (1 / len)
    points_x = []

    for i in range(num_points):
            start = np.random.uniform(start, stop, 1)
            x=np.linspace(start, subsequence * step + start, subsequence)
            for i in x:
                points_x.append(i[0])
    return np.asarray(points_x)


x=generate_points(400,500,300,6)

points1=[func1(i).numpy()[0] for i in x ]
points2=[func2(i).numpy()[0] for i in x]
print(np.asarray(points1).shape, "shape")
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
for i in range(300):
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

kernel1=ExponentialKernel(input_dims=8)
#kernel2=MaternKernel(input_dims=8)
kernel3=PolynomialKernel(2,input_dims=8)
kernel4=RationalQuadraticKernel(input_dims=8)
kernel5=SquaredExponentialKernel(input_dims=8)
kernel7=ConstantKernel(input_dims=8)
kernels=[kernel1, kernel3, kernel4,  kernel7 ]
kernel=LinearModelOfCoregionalizationKernel(kernels,output_dims= 8,input_dims=8,Q=4)

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

data=mogptk.DataSet(data1+data2+data3+data4)
print("created data set")
model=mogptk.Model(data,kernel )


model.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model.log_marginal_likelihood(),"trained")



model.save("LMC_6_full")

