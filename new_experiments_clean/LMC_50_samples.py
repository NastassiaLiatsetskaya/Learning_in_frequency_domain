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


x=generate_points(300,500,700,50)

points1=[func1(i).numpy()[0] for i in x ]
points2=[func2(i).numpy()[0] for i in x]
print(np.asarray(points1).shape, "shape")
conv_points=np.log(np.convolve(np.asarray(points1),np.asarray(points1), "same"))
points2=conv_points
f2=[]
f1=[]
i=0
while i< len(conv_points):
    f2.append(conv_points[i:i+50])
    f1.append(points1[i:i+50])
    i+=50


f1_real = []
f1_imag = []
f2_real = []
f2_imag = []
for i in range(700):
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



kernel=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_sec=LinearModelOfCoregionalizationKernel(kernels, output_dims=2,input_dims=52,Q=6)
kernel_third=LinearModelOfCoregionalizationKernel(kernels, output_dims= 2,input_dims=52,Q=6)
kernel_fourth=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_fifth=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_sixth=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_seventh=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_8=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_9=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_10=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_11=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_12=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_13=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_14=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_15=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_16=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_17=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_18=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_19=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_20=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_21=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_22=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_23=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_24=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_25=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)
kernel_26=LinearModelOfCoregionalizationKernel(kernels,output_dims= 2,input_dims=52,Q=6)










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

data12=extract_points(y_train_real,22,23,x_train_real)
model12=mogptk.Model(data12, kernel_12)

data13=extract_points(y_train_real,24,25,x_train_real)
model13=mogptk.Model(data13, kernel_13)

data14=extract_points(y_train_real,26,27,x_train_real)
model14=mogptk.Model(data14, kernel_14)

data15=extract_points(y_train_real,28,29,x_train_real)
model15=mogptk.Model(data15, kernel_15)

data16=extract_points(y_train_real,30,31,x_train_real)
model16=mogptk.Model(data16, kernel_16)

data17=extract_points(y_train_real,32,33,x_train_real)
model17=mogptk.Model(data17, kernel_17)

data18=extract_points(y_train_real,34,35,x_train_real)
model18=mogptk.Model(data18, kernel_18)

data19=extract_points(y_train_real,36,37,x_train_real)
model19=mogptk.Model(data19, kernel_19)

data20=extract_points(y_train_real,38,39,x_train_real)
model20=mogptk.Model(data20, kernel_20)

data21=extract_points(y_train_real,40,41,x_train_real)
model21=mogptk.Model(data21, kernel_21)

data22=extract_points(y_train_real,42,43,x_train_real)
model22=mogptk.Model(data22, kernel_22)

data23=extract_points(y_train_real,44,45,x_train_real)
model23=mogptk.Model(data23, kernel_23)

data24=extract_points(y_train_real,46,47,x_train_real)
model24=mogptk.Model(data24, kernel_24)

data25=extract_points(y_train_real,48,49,x_train_real)
model25=mogptk.Model(data25, kernel_25)

data26=extract_points(y_train_real,50,51,x_train_real)
model26=mogptk.Model(data26, kernel_26)



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
model12.train(method="Adam", plot=True, iters=500, error="MAE")
print(model12.log_marginal_likelihood(),"trained")
model13.train(method="Adam", plot=True, iters=500, error="MAE")
print(model13.log_marginal_likelihood(),"trained")
model14.train(method="Adam", plot=True, iters=500, error="MAE")
print(model14.log_marginal_likelihood(),"trained")
model15.train(method="Adam", plot=True, iters=500, error="MAE")
print(model15.log_marginal_likelihood(),"trained")
model16.train(method="Adam", plot=True, iters=500, error="MAE")
print(model16.log_marginal_likelihood(),"trained")
model17.train(method="Adam", plot=True, iters=500, error="MAE")
print(model17.log_marginal_likelihood(),"trained")
model18.train(method="Adam", plot=True, iters=500, error="MAE")
print(model18.log_marginal_likelihood(),"trained")
model19.train(method="Adam", plot=True, iters=500, error="MAE")
print(model19.log_marginal_likelihood(),"trained")
model20.train(method="Adam", plot=True, iters=500, error="MAE")
print(model20.log_marginal_likelihood(),"trained")
model21.train(method="Adam", plot=True, iters=500, error="MAE")
print(model21.log_marginal_likelihood(),"trained")
model22.train(method="Adam", plot=True, iters=500, error="MAE")
print(model22.log_marginal_likelihood(),"trained")
model23.train(method="Adam", plot=True, iters=500, error="MAE")
print(model23.log_marginal_likelihood(),"trained")
model24.train(method="Adam", plot=True, iters=500, error="MAE")
print(model24.log_marginal_likelihood(),"trained")
model25.train(method="Adam", plot=True, iters=500, error="MAE")
print(model25.log_marginal_likelihood(),"trained")
model26.train(method="Adam", plot=True, iters=500, error="MAE")
print(model26.log_marginal_likelihood(),"trained")











model1.save("LMC1_50")
model2.save("LMC2_50")
model3.save("LMC3_50")
model4.save("LMC4_50")
model5.save("LMC5_50")
model6.save("LMC6_50")
model7.save("LMC7_50")
model8.save("LMC8_50")
model9.save("LMC9_50")
model10.save("LMC10_50")
model11.save("LMC11_50")
model12.save("LMC12_50")
model13.save("LMC13_50")
model14.save("LMC14_50")
model15.save("LMC15_50")
model16.save("LMC16_50")
model17.save("LMC17_50")
model18.save("LMC18_50")
model19.save("LMC19_50")
model20.save("LMC20_50")
model21.save("LMC21_50")
model22.save("LMC22_50")
model23.save("LMC23_50")
model24.save("LMC24_50")
model25.save("LMC25_50")
model26.save("LMC26_50")