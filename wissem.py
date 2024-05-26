#create a tensor with constant
import tensorflow as tf
from numpy import float32

rank0tensor=tf.constant([1,1])
print(rank0tensor)
rank1tensor=tf.constant([[1,1],[2,2]],dtype=float32)
print(rank1tensor)
dim=rank1tensor.ndim
print(dim)
print(rank0tensor.shape)
randomtensor=tf.random.Generator.from_seed(42)
randomtensor=randomtensor.normal(shape=(5,300))
print(randomtensor)
rand1tf=tf.random.uniform(shape=[5,300],minval=0, maxval=1)
rand2tf=tf.random.uniform(shape=[5,300],minval=0, maxval=1)
print(rand1tf)
#multiplivation
tf3= tf.matmul(rand2tf, tf.transpose(rand1tf))
print(tf3)
tf33= tf.tensordot(rand2tf, tf.transpose(rand1tf),axes=1)
print(tf33)
tf22= tf.matmul(rand2tf, tf.reshape(rand1tf,(300,5)))
print(tf22)
rank11tf=tf.random.uniform(shape=[224, 224, 3],minval=0, maxval=1)
print(rank11tf)
max=tf.reduce_max(rank11tf)
print(max)
#check min
min=tf.reduce_min(rank11tf)
print(min)
#create a tensor using randint
tensor11=tf.random.uniform(shape=[1,224, 224, 3],minval=0, maxval=1)
#rashpe
print(tf.reshape(tensor11,[224, 224, 3]))
tensorjdid=tf.random.uniform(shape=[10],minval=0, maxval=1)
index=tf.argmax(tensorjdid)
print(tensorjdid)
print(index)
#onehotencoder
import numpy as np
tensor4=tf.constant(np.random.randint(low=0, high=100, size=10))
res=tf.one_hot(tensor4,depth=9)
print(res)



