import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #just to hide messages

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

#Initialization of Tensors
constant = tf.constant(4, shape=(1,1), dtype=tf.float32)
print(constant)

array_2d = tf.constant([[1,2,3],[4,5,6]])
print(array_2d)

ones = tf.ones((3,3))
print(ones)

zeros = tf.zeros((2,3))
print(zeros)

i = tf.eye(3)
print(i)

distribution = tf.random.normal((3,3), mean=0, stddev=1)
print(distribution)

uniform = tf.random.uniform((1,3), minval=0, maxval=1)
print(uniform)

tf_range = tf.range(start=1, limit=10, delta=2) #can simply tf.range(n)
print(tf_range)

cast_thing = tf.cast(tf_range, dtype=tf.float64) #change vector to different type
print(cast_thing)

#Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

add_tf = tf.add(x,y)
add = x+y
print(add_tf, add)

subtract_tf = tf.subtract(x,y)
subtract = x-y
print(subtract_tf, subtract)

divide_tf = tf.divide(x,y)
divide = x/y
print(divide_tf, divide)

multiply_tf = tf.multiply(x,y)
multiply = x*y
print(multiply_tf, multiply)

dot_tensor = tf.tensordot(x,y,axes=1)
dot_tensor_in_another_way = tf.reduce_sum(x*y,axis=0)
print(dot_tensor,dot_tensor_in_another_way)

power = x ** 5
print(power)

n1 = tf.random.normal((2,3))
n2 = tf.random.normal((3,4))
n1_mul_n2_tf = tf.matmul(n1,n2)
print(n1_mul_n2_tf)
n1_mul_n2 = n1 @ n2
print(n1_mul_n2)

#Indexing
vector = tf.constant([0,1,1,2,3,1,2,3])
print(vector[:])		#all
print(vector[1:])		#start from index 1
print(vector[1:3])		#start from index 1 and end BEFORE index 3
print(vector[::2])		#every 2
print(vector[3::2])		#start rom index 3 every 2
print(vector[::-1])		#reverse order

#choose specific index
indices = tf.constant([0,3])
vector_ind = tf.gather(vector,indices)
print(vector_ind)

matrix = tf.constant([[1,2],
					  [3,4],
					  [5,6]])

print(matrix[0,:])		#index 0 row, all
print(matrix[0:2,:])	#index 0 row to index 1 row, all

#Reshape
item = tf.range(9)
print(item)

item = tf.reshape(item,(3,3))
print(item)

item = tf.transpose(item, perm=[1,0])
print(item)