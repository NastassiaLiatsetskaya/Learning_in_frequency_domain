import mogptk
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

x_train=torch.tensor(np.array(x_train))
y_train=torch.tensor(np.array(f2))

method = 'BNSE'

data1=mogptk.Data(x_train ,y_train[:,0])

data2=mogptk.Data(x_train ,y_train[:,1])

data3=mogptk.Data(x_train ,y_train[:,2])

data4=mogptk.Data(x_train ,y_train[:,3])

data5=mogptk.Data(x_train ,y_train[:,4])

data6=mogptk.Data(x_train ,y_train[:,5])
data_train=mogptk.DataSet([data1,data2,data3,data4,data5,data6])
model=mogptk.MOSM(data_train,15)

model.init_parameters(method)

model.predict()
model.save('multioutput')
