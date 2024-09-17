import mogptk
from scipy.fft import rfft
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import SpectralMixtureKernel
import time
torch.manual_seed(1)

start=time.time()

def low_fidelity(x):
    return np.sin(8*np.pi*x)

def high_fidelity(x):
    return np.sin(8*np.pi*x+np.pi/10)**2+np.cos(4*np.pi*x)



x=np.linspace(0,1,1200)
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
for i in range(150):
    value1 = rfft(f1[i])
    value2 = rfft(f2[i])
    real_train=[]

    for k in value1:
        real_train.append(k.real)
        real_train.append(k.imag)
        # weights of frequencies! Which parts of high fidelity

    for p in range(len(points[i])):
        real_train.append(points[i][p])

   # real_train= np.concatenate(np.asarray(real_train), np.asarray(points[i]))
    f1_real.append(real_train)

    real_train2=[]
    imag_train2=[]
    for k in value2:
        real_train2.append(k.real)
        imag_train2.append(k.imag)

    f2_real.append(real_train2)
    f2_imag.append(imag_train2)

print(f1_real[:2], "samples")


f1_real=np.delete(f1_real,1,1)
f1_real=np.delete(f1_real, 8,1)
x_train_real = torch.tensor(np.asarray(f1_real))





y_train_real = torch.tensor(np.asarray(f2_real))
y_train_imag = torch.tensor(np.asarray(f2_imag))

method="BNSE"


kernel1_r=SpectralMixtureKernel(Q=15,input_dims=16)
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
print(data1.get_input_dims(),"input dimensions data1")
model1=mogptk.SM(data1,Q=15)

model1.init_parameters(method)


data2=mogptk.DataSet(x_train_real,y_train_real[:,1])
model2=mogptk.SM(data2,Q=15)
model2.init_parameters(method)

data3=mogptk.DataSet(x_train_real,y_train_real[:,2])
model3=mogptk.SM(data3, Q=15)
model3.init_parameters(method)

data4=mogptk.DataSet(x_train_real,y_train_real[:,3])
model4=mogptk.SM(data4,Q=15)
model4.init_parameters(method)

data5=mogptk.DataSet(x_train_real,y_train_real[:,4])
model5=mogptk.SM(data5, Q=15)
model5.init_parameters(method)

#data6=mogptk.DataSet(x_train_real,y_train_real[:,5])
#model6=mogptk.Model(data6,kernel6_r)



data1_im=mogptk.DataSet(x_train_real, y_train_imag[:,0])
print(data1_im.get_input_dims(),"input dimensions data1")
model1_im=mogptk.SM(data1_im,Q=15 )

model1_im.init_parameters(method)

data2_im=mogptk.DataSet(x_train_real,y_train_imag[:,1])
model2_im=mogptk.SM(data2_im, Q=15)
model2_im.init_parameters(method)

data3_im=mogptk.DataSet(x_train_real,y_train_imag[:,2])
model3_im=mogptk.SM(data3_im, Q=15)
model3_im.init_parameters(method)

data4_im=mogptk.DataSet(x_train_real,y_train_imag[:,3])
model4_im=mogptk.SM(data4_im, Q=15)
model4_im.init_parameters(method)

data5_im=mogptk.DataSet(x_train_real,y_train_imag[:,4])
model5_im=mogptk.SM(data5_im, Q=15)
model5_im.init_parameters(method)

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
end=time.time()

model1.save("m1_r_sin_sm_more")
model2.save("m2_r_sin_sm_more")
model3.save("m3_r_sin_sm_more")
model4.save("m4_r_sin_sm_more")
model5.save("m5_r_sin_sm_more")
#model6.save("m6_r_sin")


model1_im.save("m1_i_sin_sm_more")
model2_im.save("m2_i_sin_sm_more")
model3_im.save("m3_i_sin_sm_more")
model4_im.save("m4_i_sin_sm_more")
model5_im.save("m5_i_sin_sm_more")
#model6_im.save("m6_i_sin")
print(end-start, "bnse time")