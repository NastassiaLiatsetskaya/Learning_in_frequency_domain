

import mogptk
from scipy.fft import rfft
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel, MulKernel
from mogptk.gpr.singleoutput import LinearKernel, ExponentialKernel, RationalQuadraticKernel, MaternKernel, \
    PolynomialKernel, SquaredExponentialKernel, SpectralMixtureKernel
import time
torch.manual_seed(1)


def low_fidelity(x):
    return np.sin(8*np.pi*x)

def high_fidelity(x):
    return np.sin(8*np.pi*x+np.pi/10)**2+np.cos(4*np.pi*x)



x=np.linspace(0,1,800)
points1=[low_fidelity(i) for i in x ]
points2=[high_fidelity(i) for i in x]



f2=[]
f1=[]
i=0
points=[]
while i< len(points1):
    f2.append(points2[i:i+8])
    f1.append(points1[i:i+8])
    points.append(x[i:i+8])
    i+=8


f1_real = []
f1_imag = []
f2_real = []
f2_imag = []
for i in range(100):
    value1 = rfft(f1[i])
    value2 = rfft(f2[i])
    real_train=[]

    for k in value1:
        real_train.append(k.real)
        real_train.append(k.imag)

    for p in range(len(points[i])):
        real_train.append(points[i][p])


   # real_train= np.concatenate(np.asarray(real_train), np.asarray(points[i]))
    f1_real.append(real_train+[0,0])

    real_train2=[]
    imag_train2=[]
    for k in value2:
        real_train2.append(k.real)
        imag_train2.append(k.imag)

    f2_real.append(real_train2)
    f2_imag.append(imag_train2)

#print(f1_real[:2], "samples")
x_train_real = torch.tensor(np.asarray(f1_real))
#print(x_train_real[:2])
#x_train_real=torch.nn.functional.normalize(x_train_real, p=2,dim=1)
#print(x_train_real[:2])



y_train_real = torch.tensor(np.asarray(f2_real))
y_train_imag = torch.tensor(np.asarray(f2_imag))

kernel01=SpectralMixtureKernel(Q=15,input_dims=10, active_dims=[0,1,2,3,4,5,6,7,8,9])
kernel02=SpectralMixtureKernel(Q=15,input_dims=10, active_dims=[10,11,12,13,14,15,16,17,18,19])
kernel03=SpectralMixtureKernel(Q=15,input_dims=10, active_dims=[10,11,12,13,14,15,16,17,18,19])


  #  (

  #  SquaredExponentialKernel(input_dims=10, active_dims=[10,11,12,13,14,15,16,17,18,19]))
#kernel03=SquaredExponentialKernel(input_dims=10, active_dims=[10,11,12,13,14,15,16,17,18,19])
kernel1_r=MulKernel([kernel01,kernel02])
kernel1_r=AddKernel([kernel1_r,kernel03])


#kernel1_r=AddKernel([ExponentialKernel(input_dims=18), MaternKernel(input_dims=18),PolynomialKernel(2,input_dims=18), RationalQuadraticKernel(input_dims=18), SquaredExponentialKernel(input_dims=18)])
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



#y_r_1=torch.nn.functional.normalize(y_train_real[:,0],2,0,label="r_1")
data1=mogptk.DataSet(x_train_real,y_train_real[:,0])
model1=mogptk.Model(data1,kernel1_r)

#y_r_2=torch.nn.functional.normalize(y_train_real[:,1],2,0, label="r_2")
data2=mogptk.DataSet(x_train_real,y_train_real[:,1])
model2=mogptk.Model(data2,kernel2_r)

#y_r_3=torch.nn.functional.normalize(y_train_real[:,2],2,0, label="r_3")
data3=mogptk.DataSet(x_train_real,y_train_real[:,2])
model3=mogptk.Model(data3, kernel3_r)

#y_r_4=torch.nn.functional.normalize(y_train_real[:,3],2,0, label="r_4")
data4=mogptk.DataSet(x_train_real,y_train_real[:,3])
model4=mogptk.Model(data4,kernel4_r)

#y_r_5=torch.nn.functional.normalize(y_train_real[:,4],2,0, label="r_5")
data5=mogptk.DataSet(x_train_real,y_train_real[:,4])
model5=mogptk.Model(data5, kernel5_r)

#data6=mogptk.DataSet(x_train_real,y_train_real[:,5])
#model6=mogptk.Model(data6,kernel6_r)


#y_i_1=torch.nn.functional.normalize(y_train_imag[:,0],2,0, label="i_1")
data1_im=mogptk.DataSet(x_train_real, y_train_imag[:,0])
model1_im=mogptk.Model(data1_im, kernel1_im)

#y_i_2=torch.nn.functional.normalize(y_train_imag[:,1],2,0, label="i_2")
data2_im=mogptk.DataSet(x_train_real,y_train_imag[:,1])
model2_im=mogptk.Model(data2_im, kernel2_im)

#y_i_3=torch.nn.functional.normalize(y_train_imag[:,2],2,0, label="i_3")
data3_im=mogptk.DataSet(x_train_real,y_train_imag[:,2])
model3_im=mogptk.Model(data3_im, kernel3_im)

#y_i_4=torch.nn.functional.normalize(y_train_imag[:,3],2,0, label="i_4")
data4_im=mogptk.DataSet(x_train_real,y_train_imag[:,3])
model4_im=mogptk.Model(data4_im, kernel4_im)

#y_i_5=torch.nn.functional.normalize(y_train_imag[:,4],2,0, label="i_5")
data5_im=mogptk.DataSet(x_train_real,y_train_imag[:,4])
model5_im=mogptk.Model(data5_im, kernel5_im)

#data6_im=mogptk.DataSet(x_train_real,y_train_imag[:,5])
#model6_im=mogptk.Model(data6_im, kernel6_im)



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
#model6.train(method="Adam", plot=True, iters=500, error="MAE")
#print(model6.log_marginal_likelihood(),"trained")

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
#model6_im.train(method="Adam", plot=True, iters=500, error="MAE")
#print(model6_im.log_marginal_likelihood(), "trained")


model1.save("m1_r_sin_sm")
model2.save("m2_r_sin_sm")
model3.save("m3_r_sin_sm")
model4.save("m4_r_sin_sm")
model5.save("m5_r_sin_sm")
#model6.save("m6_r_sin")


model1_im.save("m1_i_sin_sm")
model2_im.save("m2_i_sin_sm")
model3_im.save("m3_i_sin_sm")
model4_im.save("m4_i_sin_sm")
model5_im.save("m5_i_sin_sm")
#model6_im.save("m6_i_sin")