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
    f1_real.append([i.real for i in value1])
    f1_imag.append([i.imag for i in value1])

    f2_real.append([i.real for i in value2])
    f2_imag.append([i.imag for i in value2])

x_train_real = torch.tensor(np.asarray(f1_real))

print(f1_imag[:10],"imag")

f1_imag_cleaned = np.asarray(f1_imag)[:, 1:3]
x_train_imag = torch.tensor(f1_imag_cleaned)

y_train_real = torch.tensor(np.asarray(f2_real))
y_train_imag = torch.tensor(np.asarray(f2_imag))

kernel1_r=AddKernel([ExponentialKernel(), LinearKernel(),MaternKernel(),PolynomialKernel(2), RationalQuadraticKernel(), ExponentialKernel()])
kernel2_r=AddKernel([ExponentialKernel(), LinearKernel(),MaternKernel(),PolynomialKernel(2),RationalQuadraticKernel(),ExponentialKernel()])
kernel3_r=AddKernel([ExponentialKernel(), LinearKernel(),MaternKernel(),PolynomialKernel(2),RationalQuadraticKernel(),ExponentialKernel()])
kernel4_r=AddKernel([ExponentialKernel(), LinearKernel(),MaternKernel(),PolynomialKernel(2),RationalQuadraticKernel(),ExponentialKernel()])
kernel1_im=AddKernel([ExponentialKernel(),LinearKernel(),MaternKernel(),PolynomialKernel(2),ExponentialKernel(),RationalQuadraticKernel()])
kernel2_im=AddKernel([ExponentialKernel(), LinearKernel(),MaternKernel(),PolynomialKernel(2), RationalQuadraticKernel(),ExponentialKernel()])
kernel3_im=AddKernel([ExponentialKernel(), LinearKernel(),MaternKernel(),PolynomialKernel(2),RationalQuadraticKernel(), ExponentialKernel()])
kernel4_im=AddKernel([ExponentialKernel(), LinearKernel(),MaternKernel(),PolynomialKernel(2), RationalQuadraticKernel(),ExponentialKernel()])

print(x_train_real, "x train")

data1=mogptk.DataSet(x_train_real,y_train_real[:,0])
model1=mogptk.Model(data1,kernel1_r)
data2=mogptk.DataSet(x_train_real,y_train_real[:,1])
model2=mogptk.Model(data2,kernel2_r)
data3=mogptk.DataSet(x_train_real,y_train_real[:,2])
model3=mogptk.Model(data3, kernel3_r)
data4=mogptk.DataSet(x_train_real,y_train_real[:,3])
model4=mogptk.Model(data4,kernel4_r)

data1_im=mogptk.DataSet(x_train_imag, y_train_imag[:,0])
model1_im=mogptk.Model(data1_im, kernel1_im)
data2_im=mogptk.DataSet(x_train_imag,y_train_imag[:,1])
model2_im=mogptk.Model(data2_im, kernel2_im)
data3_im=mogptk.DataSet(x_train_imag,y_train_imag[:,2])
model3_im=mogptk.Model(data3_im, kernel3_im)
data4_im=mogptk.DataSet(x_train_imag,y_train_imag[:,3])
model4_im=mogptk.Model(data4_im, kernel4_im)



model1.train(method="Adam", plot=True, iters=700,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")
model2.train(method="Adam", plot=True, iters=700,  error="MAE")
print(model2.log_marginal_likelihood(),"trained")
model3.train(method="Adam", plot=True, iters=700,  error="MAE")
print(model3.log_marginal_likelihood(),"trained")
model4.train(method="Adam", plot=True, iters=700, error="MAE")
print(model4.log_marginal_likelihood(),"trained")


model1_im.train(method="Adam", plot=True, iters=700,  error="MAE")
print(model1_im.log_marginal_likelihood(),"trained")
model2_im.train(method="Adam", plot=True, iters=700,  error="MAE")
print(model2_im.log_marginal_likelihood(),"trained")
model3_im.train(method="Adam", plot=True, iters=700,  error="MAE")
print(model3_im.log_marginal_likelihood(),"trained")
model4_im.train(method="Adam", plot=True, iters=700, error="MAE")
print(model4_im.log_marginal_likelihood(), "trained")



model1.save("first_real_compl_conv_custom2_1")
model2.save("second_real_compl_conv_custom2_1")
model3.save("third_real_compl_conv_custom2_1")
model4.save("fourth_real_compl_conv_custom2_1")
model1_im.save("first_im_compl_conv_custom2_1")
model2_im.save("second_im_compl_conv_custom2_1")
model3_im.save("third_im_compl_conv_custom2_1")
model4_im.save("fourth_im_compl_conv_custom2_1")