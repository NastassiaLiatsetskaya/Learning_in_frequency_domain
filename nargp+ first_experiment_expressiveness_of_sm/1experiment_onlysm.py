import mogptk
import numpy as np
import torch
from mogptk import Data, DataSet
torch.manual_seed(1)
import json
import time

start_time = time.time()
def high_fidelity(x):
    low_freq= np.sin(np.pi*x)+2*np.sin(2*np.pi*x)+ np.sin(3*np.pi*x)
    high_freq=np.sin(7*np.pi*x)+2*np.sin(8*np.pi*x)+ np.sin(9*np.pi*x)
    return low_freq+high_freq

num_training_samples=400
length=6

x_train=np.linspace(0,3,2400)
y_train=np.asarray([high_fidelity(i) for i in x_train])


method="BNSE"
data1=mogptk.Data(x_train ,y_train)
model1=mogptk.SM(DataSet(data1), Q=15)

model1.init_parameters(method)



model1.train(method="Adam", iters=500, error="MAE")
print("trained")

end_time = time.time()
model1.save("1experiment")
print(end_time-start_time, "time")
#BNSE failed on two processes