import numpy as np
import scipy.io
import math
import csv
from matplotlib import pyplot as plt
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,scale, normalize
from numpy import corrcoef, sum, log, arange
from numpy.random import rand
from pylab import pcolor, show, colorbar, xticks, yticks
import csv
import plotly.plotly as py
import plotly.figure_factory as FF
from sklearn.utils import shuffle
from sklearn.feature_extraction import DictVectorizer
import sys
import pdb

data = scipy.io.loadmat('letters_data.mat')
train = data['train_x']
train = scale(train)
train = normalize(train)
labels = data['train_y']
oneHotEncoder = OneHotEncoder()
oneHotEncoder.fit(labels)
labels = oneHotEncoder.transform(labels).toarray()
X_train, X_val, y_train, y_val = train_test_split(train, labels, test_size=0.2, random_state=42)


def tanh(x):
	a = np.exp(x)-np.exp(-x)
	b = np.exp(x)+np.exp(-x)
	return np.divide(a,b)

def tanh_prime(x):
	return np.divide(4.,np.square(np.exp(-x)+np.exp(x)))

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def softplus(x):
	return np.log(1.+np.exp(x))

def complex_exponential(x):
	return np.exp(1j*x)

def complex_exponential_prime(x):
	return 1j*np.exp(x)


def sigmoid_prime(x):
	return np.dot(sigmoid(x),(1.-sigmoid(x)))

def computeHiddenLayer(inp,V):
	v = np.dot(V,inp)
	v = v.reshape(v.shape[0])
	t = tanh(v)
	ret = t.reshape(t.shape[0],1)
	return ret

def computeOutputLayer(hidden,W):
	v = np.dot(W,hidden)
	v = v.reshape(v.shape[0])
	sig = sigmoid(v)
	ret = sig.reshape(sig.shape[0],1)
	return ret

def loss(y,z):
	return np.square(np.absolute(z-y))

def loss_grad_z(y,z):
	return 2.*(z-y)

def log_loss(y,z):
	a = 0.
	for i in range(len(y)):
		a = a - np.dot( y[i] , np.log(z[i]) ) - np.dot( (1-y[i]), np.log(1-z[i]) )
	return a

def log_loss_grad_z(y,z):
	return -np.divide(y,z)+np.divide((1-y),(1-z))

def loss_grad_h(z,W,loss_grad_z):
	return np.dot(np.multiply(np.multiply(z,1.-z),loss_grad_z).T,W).T

def initialize_weights(x,y):
	return np.array([np.random.normal(0,1./np.sqrt(y)) for i in range(x*y)]).reshape(x,y)




def stochasticGradientDescent(epochs, learning_rate_V, learning_rate_W, decay):
	V = initialize_weights(200,785)
	W = initialize_weights(26,201)
	for j in range(epochs):
		for i in range(len(X_train)):
			input_vector = X_train[i]
			input_vector = np.append(input_vector,1) # add bias term
			input_vector = input_vector.reshape(input_vector.shape[0],1)
			label = y_train[i]
			label = label.reshape(label.shape[0],1)
			hidden = computeHiddenLayer(input_vector,V)
			hidden = np.append(hidden,1) # add bias term
			hidden = hidden.reshape(hidden.shape[0],1)
			z = computeOutputLayer(hidden,W)
			L = log_loss(label,z)
			L_grad_z = log_loss_grad_z(label,z)
			v_0 = np.multiply(np.multiply(L_grad_z,z),1.-z)
			L_grad_w = np.dot(v_0,hidden.T)
			L_grad_h = loss_grad_h(z,W,L_grad_z)
			v_1 = np.multiply(L_grad_h,tanh_prime(hidden))
			L_grad_v = np.dot(v_1,input_vector.T)
			L_grad_v = np.delete(L_grad_v,200,0)# remove bias
			W = W - learning_rate_V*L_grad_w
			V = V - learning_rate_W*L_grad_v

		learning_rate_V = learning_rate_V *1./(1.+decay*j)
		learning_rate_W = learning_rate_W *1./(1.+decay*j)
	return V,W

def compute_hidden_batch(inp,V):
	v = np.dot(inp,V.T)
	t = tanh(v)
	return t

def compute_output_batch(hiddens,W):
	v = np.dot(W,hiddens.T).T
	sig = sigmoid(v)
	return sig

def log_loss_batch(y,z,batch_size):
	a = -np.sum(y*np.log(z),axis=1)# element wise dot product
	b = -np.sum((1.-y)*np.log(1.-z),axis=1)
	res = a+b
	res = res.reshape(res.shape[0],1)
	return (np.sum(res,axis=0)/batch_size)[0]

def log_loss_grad_z_batch(labels,outputs,batch_size):
	return log_loss_grad_z(labels,outputs)

def loss_grad_h_batch(z,W,L_grad_z,batch_size):
	return np.dot(np.multiply(np.multiply(z,1.-z),L_grad_z),W)

def loss_grad_w_batch(v_0,hiddens,batch_size):
	res = np.dot(v_0.T,hiddens)/batch_size
	return res

def loss_grad_v_batch(v_1,input_matrix,batch_size):
	res = np.dot(v_1.T,input_matrix)/batch_size
	return res



def mini_batch_gradient_descent(epochs, batch_size, learning_rate_V, learning_rate_W, decay):
	V = initialize_weights(200,785)
	W = initialize_weights(26,201)
	for j in range(epochs):
		for i in range(len(X_train)/batch_size):
			input_matrix = X_train[i*batch_size:(i+1)*batch_size]
			bias = np.ones((batch_size,1))
			input_matrix = np.append(input_matrix,bias,1)
			labels = y_train[i*batch_size:(i+1)*batch_size]
			hiddens = compute_hidden_batch(input_matrix,V)
			hiddens = np.append(hiddens,bias,1)
			outputs = compute_output_batch(hiddens,W)
			L = log_loss_batch(labels,outputs,batch_size)#
			if(j==0):
				loss_values.append(L)
			L_grad_outputs = log_loss_grad_z_batch(labels,outputs,batch_size)
			v_0 = np.multiply(np.multiply(L_grad_outputs,outputs),1.-outputs)
			L_grad_w = loss_grad_w_batch(v_0,hiddens,batch_size)
			L_grad_h = loss_grad_h_batch(outputs,W,L_grad_outputs,batch_size)
			v_1 = np.multiply(L_grad_h,tanh_prime(hiddens))
			L_grad_v = loss_grad_v_batch(v_1,input_matrix,batch_size)
			L_grad_v = np.delete(L_grad_v,200,0)# remove bias
			W = W - learning_rate_V*L_grad_w
			V = V - learning_rate_W*L_grad_v

		learning_rate_V = learning_rate_V *1./(1.+decay*j)
		learning_rate_W = learning_rate_W *1./(1.+decay*j)
	return V,W


def sin_mini_batch_gradient_descent(data,epochs, batch_size, learning_rate_V, learning_rate_W, decay):
	V = initialize_weights(200,1001)
	W = initialize_weights(1,201)
	for j in range(epochs):
		for i in range(len(data)/batch_size):
			input_matrix = X_train[i*batch_size:(i+1)*batch_size]
			bias = np.ones((batch_size,1))
			input_matrix = np.append(input_matrix,bias,1)
			labels = y_train[i*batch_size:(i+1)*batch_size]
			hiddens = compute_hidden_batch(input_matrix,V)
			hiddens = np.append(hiddens,bias,1)
			outputs = compute_output_batch(hiddens,W)
			L = log_loss_batch(labels,outputs,batch_size)#
			if(j==0):
				loss_values.append(L)
			L_grad_outputs = log_loss_grad_z_batch(labels,outputs,batch_size)
			v_0 = np.multiply(np.multiply(L_grad_outputs,outputs),1.-outputs)
			L_grad_w = loss_grad_w_batch(v_0,hiddens,batch_size)
			L_grad_h = loss_grad_h_batch(outputs,W,L_grad_outputs,batch_size)
			v_1 = np.multiply(L_grad_h,tanh_prime(hiddens))
			L_grad_v = loss_grad_v_batch(v_1,input_matrix,batch_size)
			L_grad_v = np.delete(L_grad_v,200,0)# remove bias
			W = W - learning_rate_V*L_grad_w
			V = V - learning_rate_W*L_grad_v

		learning_rate_V = learning_rate_V *1./(1.+decay*j)
		learning_rate_W = learning_rate_W *1./(1.+decay*j)
	return V,W


def predict(images, V, W):
	ret = []
	for i in range(len(images)):
		image = np.append(images[i],1)
		image = image.reshape(image.shape[0],1.)
		hidden = computeHiddenLayer(image,V)
		hidden = np.append(hidden,1)
		hidden = hidden.reshape(hidden.shape[0],1.)
		output = computeOutputLayer(hidden,W)
		ret.append(output)
	return ret

batch_size = 50
epochs = 1
batches = [i for i in range(len(X_train)/batch_size)]
loss_values = []

V,W = mini_batch_gradient_descent(epochs,batch_size,0.1,0.2,0.001)
predict_train = predict(X_train,V,W)
predict_val = predict(X_val,V,W)
#V,W = stochasticGradientDescent(10,0.1,0.5,0.001)
#out = predict(X_train,V,W)
test_data = data['test_x']
test_data = scale(test_data)
test_data = normalize(test_data)
predict_test = predict(test_data,V,W)


num_correct_train=0
num_correct_val=0
training_accuracy = 0
for i in range(len(predict_train)):
	m = max(predict_train[i])
	v = [1*int(val==m) for val in predict_train[i]]
	b = v - y_train[i]
	if max(b)==0 and min(b)==0:
		num_correct_train+=1
for i in range(len(predict_val)):
	m = max(predict_val[i])
	v = [1*int(val==m) for val in predict_val[i]]
	b = v - y_val[i]
	if max(b)==0 and min(b)==0:
		num_correct_val+=1

training_accuracy = float(num_correct_train)/len(X_train)
validation_accuracy = float(num_correct_val)/len(X_val)

correctly_classified = []
incorrectly_classified = []
i = 0
while(len(correctly_classified)<5 or len(incorrectly_classified)<5):
	label = y_val[i]
	m = max(predict_val[i])
	v = [1*int(val==m) for val in predict_val[i]]
	b = v - label
	if max(b)==0 and min(b)==0:
		if(len(correctly_classified)<5):
			correctly_classified.append((X_val[i],y_val[i]))
	else:
		if(len(incorrectly_classified)<5):
			incorrectly_classified.append( (X_val[i],predict_val[i],y_val[i]) )
	i+=1

for (row,label) in correctly_classified:
	img = row.reshape(28,28)
	img = plt.imshow(img,cmap='Greys')
	index = np.argmax(label)
	plt.title("correctly labeled "+chr(ord('a')+index))
	plt.show()

for (row,prediction,label) in incorrectly_classified:
	img = row.reshape(28,28)
	img = plt.imshow(img,cmap='Greys')
	index = np.argmax(label)
	ind_pred = np.argmax(prediction)
	plt.title("incorrectly labeled "+chr(ord('a')+ind_pred)+" actually "+chr(ord('a')+index))
	plt.show()
	
plt.scatter(batches,loss_values)
axes = plt.gca()
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.title("Loss vs Iterations")















