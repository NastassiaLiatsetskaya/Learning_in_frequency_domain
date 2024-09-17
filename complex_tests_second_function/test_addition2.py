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


x = np.linspace(-3, 3, 3000)
points1 = [low_fidelity(i) for i in x]
points4 = np.asarray([200 * i for i in x])
conv_points = np.convolve(points1, points1, mode='full')
points2 = conv_points[1500:4500] + points4
vals = []
for i, k,m in zip(points2, points4,x):
    vals.append(i / 20 + k + np.sqrt(np.abs(k + i+m**12)))


conv_points=vals




f2=[]
f1=[]
i=0
points=[]
while i< len(conv_points):
    f2.append(conv_points[i:i+10])
    f1.append(points1[i:i+10])
    points.append(x[i:i+10])
    i+=10


f1_real = []
f1_imag = []
f2_real = []
f2_imag = []
for i in range(300):
    value1 = rfft(f1[i])
    value2 = rfft(f2[i])
    real_train=[]

    real_train.append(points[i][0])
    f1_real.append(real_train)
    real_train2=[]
    imag_train2=[]
    for k in value2:
        real_train2.append(k.real)
        imag_train2.append(k.imag)

    f2_real.append(real_train2)
    f2_imag.append(imag_train2)

x_train_real = torch.tensor(np.asarray(f1_real))

print(f1_imag[:10],"imag")



y_train_real = torch.tensor(np.asarray(f2_real))
y_train_imag = torch.tensor(np.asarray(f2_imag))

kernel1_r=AddKernel([ExponentialKernel(input_dims=1), MaternKernel(input_dims=1),PolynomialKernel(2,input_dims=1), RationalQuadraticKernel(input_dims=1), SquaredExponentialKernel(input_dims=1)])
kernel2_r=kernel1_r.clone()
kernel3_r=kernel1_r.clone()
kernel4_r=kernel1_r.clone()
kernel5_r=kernel1_r.clone()
kernel6_r=kernel1_r.clone()
kernel1_im=kernel1_r.clone()
kernel2_im= kernel1_r.clone()
kernel3_im= kernel1_r.clone()
kernel4_im=kernel1_r.clone()
kernel5_im=kernel1_r.clone()
kernel6_im=kernel1_r.clone()




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



data1_im=mogptk.DataSet(x_train_real, y_train_imag[:,0])
model1_im=mogptk.Model(data1_im, kernel1_im)

data2_im=mogptk.DataSet(x_train_real,y_train_imag[:,1])
model2_im=mogptk.Model(data2_im, kernel2_im)

data3_im=mogptk.DataSet(x_train_real,y_train_imag[:,2])
model3_im=mogptk.Model(data3_im, kernel3_im)

data4_im=mogptk.DataSet(x_train_real,y_train_imag[:,3])
model4_im=mogptk.Model(data4_im, kernel4_im)

data5_im=mogptk.DataSet(x_train_real,y_train_imag[:,4])
model5_im=mogptk.Model(data5_im, kernel5_im)

data6_im=mogptk.DataSet(x_train_real,y_train_imag[:,5])
model6_im=mogptk.Model(data6_im, kernel6_im)



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

model1_im.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model1_im.log_marginal_likelihood(),"trained")
model2_im.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model2_im.log_marginal_likelihood(),"trained")
model3_im.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model3_im.log_marginal_likelihood(),"trained")
model4_im.train(method="Adam", plot=True, iters=500, error="MAE")
print(model4_im.log_marginal_likelihood(), "trained")
model5_im.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model5_im.log_marginal_likelihood(),"trained")
model6_im.train(method="Adam", plot=True, iters=500, error="MAE")
print(model6_im.log_marginal_likelihood(), "trained")


model1.save("m1_r_convo_p2")
model2.save("m2_r_convo_p2")
model3.save("m3_r_convo_p2")
model4.save("m4_r_convo_p2")
model5.save("m5_r_convo_p2")
model6.save("m6_r_convo_p2")


model1_im.save("m1_i_convo_p2")
model2_im.save("m2_i_convo_p2")
model3_im.save("m3_i_convo_p2")
model4_im.save("m4_i_convo_p2")
model5_im.save("m5_i_convo_p2")
model6_im.save("m6_i_convo_p2")