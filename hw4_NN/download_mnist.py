import tensorflow as tf
import scipy.misc

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

for i in range(len(x_train)):
    scipy.misc.imsave(str(y_train[i])+'/'+str(i)+'.jpg', x_train[i])


print("hello")