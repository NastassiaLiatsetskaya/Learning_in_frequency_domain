import mogptk
import numpy as np
import torch
from mogptk import Data, DataSet
torch.manual_seed(1)
import json

def func1(t):
    return torch.tensor([np.sin(8*np.pi*t)])

def func3(t):
    return torch.tensor([np.sin(8*np.pi*t+np.pi/10)**2 + np.cos(4*np.pi*t)])

num_training_samples=400
length=6

x_train=[]
for i in range(num_training_samples):
    index=i+i*6
    x_train.append(np.linspace(index/2799, (index+6)/2799, 6))


f1=[]
f2=[]
for i in x_train:
    values1=[]
    values2=[]
    for index in i:
        values1.append(func1(index).item())
        values2.append(func3(index).item())
    f1.append(values1)
    f2.append(values2)

x_train=torch.tensor(np.array(x_train))
y_train=torch.tensor(np.array(f2))

method = 'BNSE'

data1=mogptk.Data(x_train ,y_train[:,0])
model1=mogptk.SM(DataSet(data1),15)

data2=mogptk.Data(x_train ,y_train[:,1])
model2=mogptk.SM(DataSet(data2),15)

data3=mogptk.Data(x_train ,y_train[:,2])
model3=mogptk.SM(DataSet(data3),15)

data4=mogptk.Data(x_train ,y_train[:,3])
model4=mogptk.SM(DataSet(data4),15)

data5=mogptk.Data(x_train ,y_train[:,4])
model5=mogptk.SM(DataSet(data5),15)

data6=mogptk.Data(x_train ,y_train[:,5])
model6=mogptk.SM(DataSet(data6),15)

model1.init_parameters(method)
model2.init_parameters(method)
model3.init_parameters(method)
model4.init_parameters(method)
model5.init_parameters(method)
model6.init_parameters(method)

model1.train(method="Adam", iters=500, error="MAE")
print("trained")
model2.train(method="Adam", iters=500, error="MAE")
print("trained")
model3.train(method="Adam", iters=500, error="MAE")
print("trained")
model4.train(method="Adam", iters=500,error="MAE")
print("trained")
model5.train(method="Adam", iters=500,error="MAE")
print("trained")
model6.train(method="Adam", iters=500,error="MAE")

model1.save("f_independent_sm3")
model2.save("s_independent_sm3")
model3.save("t_independent_sm3")
model4.save("f1_independent_sm3")
model5.save("ff_independent_sm3")
model6.save("si_independent_sm3")
#BNSE failed on two processes