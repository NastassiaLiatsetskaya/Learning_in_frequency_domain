import json
import torch
import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import LinearKernel, ExponentialKernel, RationalQuadraticKernel, MaternKernel, \
    PolynomialKernel, SquaredExponentialKernel

torch.manual_seed(1)


def low_fidelity(x):
    return np.sin(np.pi * x)


x = np.linspace(-3, 3, 5000)
points1 = [low_fidelity(i) for i in x]
points4 = np.asarray([(200 * i) for i in x])
conv_points = np.convolve(points1, points1, mode='full')
points2 = conv_points[2500:7500] + points4
vals = []
for i, k, m in zip(points2, points4, x):
    vals.append((i / 20 + k / 3 + np.sqrt(np.abs(k / 60 + i))) / 10)



conv_points=vals




f2=[]

i=0
points=[]
while i< len(conv_points):
    f2.append(conv_points[i:i+10])

    points.append(x[i:i+10])
    i+=10


f1_real = []

f2_real = []

for i in range(500):

    value2 = (f2[i])

    f1_real.append(points[i])


    f2_real.append(value2)


x_train_real = torch.tensor(np.asarray(f1_real))




y_train_real = torch.tensor(np.asarray(f2_real))


kernel1_r=AddKernel([ExponentialKernel(input_dims=10), MaternKernel(input_dims=10),PolynomialKernel(2,input_dims=10), RationalQuadraticKernel(input_dims=10), SquaredExponentialKernel(input_dims=10)])
kernel2_r=kernel1_r.clone()
kernel3_r=kernel1_r.clone()
kernel4_r=kernel1_r.clone()
kernel5_r=kernel1_r.clone()
kernel6_r=kernel1_r.clone()
kernel7_r=kernel1_r.clone()
kernel8_r=kernel1_r.clone()
kernel9_r=kernel1_r.clone()
kernel10_r=kernel1_r.clone()





data1=mogptk.DataSet(x_train_real,y_train_real[:,0])
model1=mogptk.Model(data1,kernel1_r)

data2=mogptk.DataSet(x_train_real,y_train_real[:,1])
model2=mogptk.Model(data2,kernel2_r)

data3=mogptk.DataSet(x_train_real,y_train_real[:,2])
model3=mogptk.Model(data3, kernel3_r)

data4=mogptk.DataSet(x_train_real,y_train_real[:,3])
model4=mogptk.Model(data4,kernel4_r)

data5=mogptk.DataSet(x_train_real,y_train_real[:,4])
model5=mogptk.Model(data5, kernel5_r)

data6=mogptk.DataSet(x_train_real,y_train_real[:,5])
model6=mogptk.Model(data6,kernel6_r)

data7=mogptk.DataSet(x_train_real,y_train_real[:,6])
model7=mogptk.Model(data7,kernel7_r)

data8=mogptk.DataSet(x_train_real,y_train_real[:,7])
model8=mogptk.Model(data8, kernel8_r)

data9=mogptk.DataSet(x_train_real,y_train_real[:,8])
model9=mogptk.Model(data9,kernel9_r)

data10=mogptk.DataSet(x_train_real,y_train_real[:,9])
model10=mogptk.Model(data10,kernel10_r)





model1.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")
model2.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model2.log_marginal_likelihood(),"trained")
model3.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model3.log_marginal_likelihood(),"trained")
model4.train(method="Adam", plot=True, iters=500, error="MAE")
print(model4.log_marginal_likelihood(),"trained")
model5.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model5.log_marginal_likelihood(),"trained")
model6.train(method="Adam", plot=True, iters=500, error="MAE")
print(model6.log_marginal_likelihood(),"trained")
model7.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model7.log_marginal_likelihood(),"trained")
model8.train(method="Adam", plot=True, iters=500, error="MAE")
print(model8.log_marginal_likelihood(),"trained")
model9.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model9.log_marginal_likelihood(),"trained")
model10.train(method="Adam", plot=True, iters=500, error="MAE")
print(model10.log_marginal_likelihood(),"trained")


model1.save("m1_r_convo_p6")
model2.save("m2_r_convo_p6")
model3.save("m3_r_convo_p6")
model4.save("m4_r_convo_p6")
model5.save("m5_r_convo_p6")
model6.save("m6_r_convo_p6")

model7.save("m7_r_convo_p6")
model8.save("m8_r_convo_p6")
model9.save("m9_r_convo_p6")
model10.save("m10_r_convo_p6")