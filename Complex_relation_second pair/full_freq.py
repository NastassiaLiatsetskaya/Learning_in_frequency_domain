
import mogptk
from scipy.fft import rfft
import numpy as np
import torch
from mogptk.gpr.kernel import AddKernel
from mogptk.gpr.singleoutput import LinearKernel, ExponentialKernel, RationalQuadraticKernel, MaternKernel, \
    PolynomialKernel, SquaredExponentialKernel
from mogptk.gpr.multioutput import IndependentMultiOutputKernel

torch.manual_seed(1)


def low_fidelity(x):
    return np.sin(8*np.pi*x)

def high_fidelity(x):
    return np.sin(8*np.pi*x)**2



def generate_points(start,stop,num_points,subsequence):
    step = 0.01
    points_x = []
    for i in range(num_points):
        start = np.random.uniform(start, stop, 1)
        x = []

        for i in range(subsequence):
            x.append(start + step * i)

        for i in x:
            points_x.append(i[0])
    return np.asarray(points_x)


x=generate_points(0,1,20,60)

points1=[low_fidelity(i) for i in x ]
points2=[high_fidelity(i) for i in x]



f2=[]
f1=[]
i=0
points=[]
while i< len(points1):
    f2.append(points2[i:i+60])
    f1.append(points1[i:i+60])
    points.append(x[i:i+60])
    i+=60


f1_real = []
f1_imag = []
f2_real = []
f2_imag = []
for i in range(20):
    value1 = rfft(f1[i])
    value2 = rfft(f2[i])
    real_train=[]

    for k in value1:
        real_train.append(k.real)
        real_train.append(k.imag)

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
x_train_real = torch.tensor(np.asarray(f1_real))



def form_data_set(x_tr, y_r, y_i, dims):
    data=[]
    for i in range(dims):
        data.append(mogptk.Data(x_tr, y_r[:,i]))
        data.append(mogptk.Data(x_tr,y_i[:,i]))
    return mogptk.DataSet(data)


y_train_real = torch.tensor(np.asarray(f2_real))
y_train_imag = torch.tensor(np.asarray(f2_imag))

kernel1_r=AddKernel([ExponentialKernel(input_dims=122), MaternKernel(input_dims=122),
                     PolynomialKernel(2,input_dims=122), RationalQuadraticKernel(input_dims=122),
                     SquaredExponentialKernel(input_dims=122)])
kernels=[kernel1_r]
for i in range(61):

    kernels.append(AddKernel([ExponentialKernel(input_dims=122), MaternKernel(input_dims=122),
                     PolynomialKernel(2,input_dims=122), RationalQuadraticKernel(input_dims=122),
                     SquaredExponentialKernel(input_dims=122)]))
print(len(kernels))
kernel_full=IndependentMultiOutputKernel(kernels,output_dims=62)

data=form_data_set(x_train_real, y_train_real, y_train_imag, 31)


#data1=mogptk.DataSet(x_train_real,y_train_real[:,0])
model1=mogptk.Model(data,kernel1_r)

model1.train(method="Adam", plot=True, iters=500,  error="MAE")
print(model1.log_marginal_likelihood(),"trained")


model1.save("m1_r_sin100")
