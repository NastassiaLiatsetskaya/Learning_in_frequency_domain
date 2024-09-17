import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
torch.manual_seed(1)
import json
def func1(t):
    return torch.tensor([np.sin(t)+ np.cos(t**2)+ max(30,t) ])

def func2(t):
    return (torch.relu(func1(t))+ torch.sigmoid(func1(t)))


num_training_samples=400
length=6

x_train=[]
for i in range(num_training_samples):
    index=i+i*6
    x_train.append(np.linspace(index, index+6, 6))


print(x_train[398], "training", x_train[399])

f1=[]
f2=[]
for i in x_train:
    values1=[]
    values2=[]
    for index in i:
        values1.append(func1(index).item())
        values2.append(func2(index).item())
    f1.append(values1)
    f2.append(values2)
f1_real=[]
f1_imag=[]
f2_real=[]
f2_imag=[]
for i in range(num_training_samples):
    value1=rfft(f1[i])
    value2=rfft(f2[i])
    f1_real.append([i.real for i in value1])
    f1_imag.append([i.imag for i in value1])

    f2_real.append([i.real for i in value2])
    f2_imag.append([i.imag for i in value2])





x_train_real=torch.tensor(np.asarray(f1_real))

f1_imag_cleaned=np.asarray(f1_imag)[:,1:3]
x_train_imag=torch.tensor(f1_imag_cleaned )

y_train_real=torch.tensor(np.asarray(f2_real))
y_train_imag=torch.tensor(np.asarray(f2_imag))



data1=mogptk.DataSet(x_train_real,y_train_real[:,0])
model1=mogptk.SM(data1,Q=20)
data2=mogptk.DataSet(x_train_real,y_train_real[:,1])
model2=mogptk.SM(data2,Q=15)
data3=mogptk.DataSet(x_train_real,y_train_real[:,2])
model3=mogptk.SM(data3,Q=15)
data4=mogptk.DataSet(x_train_real,y_train_real[:,3])
model4=mogptk.SM(data4,Q=15)

data1_im=mogptk.DataSet(x_train_imag, y_train_imag[:,0])
model1_im=mogptk.SM(data1_im,Q=20)
data2_im=mogptk.DataSet(x_train_imag,y_train_imag[:,1])
model2_im=mogptk.SM(data2_im, Q=15)
data3_im=mogptk.DataSet(x_train_imag,y_train_imag[:,2])
model3_im=mogptk.SM(data3_im, Q=15)
data4_im=mogptk.DataSet(x_train_imag,y_train_imag[:,3])
model4_im=mogptk.SM(data4_im,Q=15)



model1.train(method="Adam", plot=True, iters=1500,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")
model2.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model2.log_marginal_likelihood(),"trained")
model3.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model3.log_marginal_likelihood(),"trained")
model4.train(method="Adam", plot=True, iters=500, error="MAE")
print(model4.log_marginal_likelihood(),"trained")


model1_im.train(method="Adam", plot=True, iters=1500,  error="MAE")
print(model1_im.log_marginal_likelihood(),"trained")
model2_im.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model2_im.log_marginal_likelihood(),"trained")
model3_im.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model3_im.log_marginal_likelihood(),"trained")
model4_im.train(method="Adam", plot=True, iters=500, error="MAE")
print(model4_im.log_marginal_likelihood(), "trained")



model1.save("first_real")
model2.save("second_real")
model3.save("third_real")
model4.save("fourth_real")
model1_im.save("first_im")
model2_im.save("second_im")
model3_im.save("third_im")
model4_im.save("fourth_im")

