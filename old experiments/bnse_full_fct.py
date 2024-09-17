import mogptk
from scipy.fft import rfft, rfftfreq
import numpy as np
import torch
torch.manual_seed(1)
import json
def func1(t):
    return torch.tensor([np.sin(t)+ np.cos(t**2)+ max(30,t)])
   # return t*2
def func2(t):
    return (torch.relu(func1(t))+ torch.sigmoid(func1(t)))
    #return t*3

num_training_samples=400
length=5

x_train=[]
y_train=[]
for i in range(num_training_samples):
    index=i+i*5
    x_train.append(np.linspace(index, index+5, 5))
    y_train.append(index)


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
#x=np.linspace(0,450, 400)
#f1=[func1(index).item() for index in x]
#f2=[func2(index).item() for index in y_train]


x_train=torch.tensor(f1)
print(x_train.shape, "input shape")
y_train=torch.tensor(f2)
print(y_train.shape, "output shape")

x_test=[411,412,413,414,415]
values=[func1(x).item() for x in x_test]
values_second=[func2(x) for x in x_test]
print(values, "first values")
print(values_second, "second values")
print(rfft(values_second), "output")

method="BNSE"

data1=mogptk.DataSet(x_train,y_train[:,0])
model1=mogptk.SM(data1,Q=20)
model1.init_parameters(method)
print("first")
data2=mogptk.DataSet(x_train,y_train[:,1])
model2=mogptk.SM(data2,Q=15)
model2.init_parameters(method)
print("second")
data3=mogptk.DataSet(x_train,y_train[:,2])
model3=mogptk.SM(data3,Q=15)
model3.init_parameters(method)
print("third")




#model1.train(method="Adam", plot=True, iters=500,  error="MAE")
#print(model1.log_marginal_likelihood(),"trained")
#model2.train(method="Adam", plot=True, iters=500,  error="MAE")
#print(model2.log_marginal_likelihood(),"trained")
#model3.train(method="Adam", plot=True, iters=500,  error="MAE")
#(model3.log_marginal_likelihood(),"trained")

#model1.save("first_sm_time_bnse")
#model2.save("second_sm_time_bnse")
#model3.save("third_sm_time_bnse")


x_test=torch.tensor([values])
print((x_test.shape) , "shape")
#x, y, low_bound, upper_bound=model1.predict(np.asarray(x_test))
#x1, y1, low_bound1, upper_bound1=model2.predict(np.asarray(x_test))
#x2, y2, low_bound2, upper_bound2=model3.predict(np.asarray(x_test))
##rint(y,y1,y2,"y first")
#print(low_bound,"low_bound")
#print(upper_bound,"upper_bound")
#errors={"first":model1.log_marginal_likelihood(),"second":model2.log_marginal_likelihood(),
 #       "third":model3.log_marginal_likelihood()}
#values={"first":y.tolist(), "second":y1.tolist(), "third":y2.tolist(),
 #       "true1":values_second[0].item() , "true2":values_second[1].item() ,
 #          "true3":values_second[2].item() ,"true4":values_second[3].item() ,"true5":values_second[4].item()}
#with open("results_sm_time_bnse.json", "a") as f:
#    json.dump(values,f)