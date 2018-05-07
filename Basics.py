import tensorflow as tf

#Creamos dos tensores

x1 = tf.constant([1,2,3,4,5])
x2 = tf.constant([6,7,8,9,10])
res = tf.multiply(x1,x2)

#print (res)

sess = tf.Session()
print(sess.run(res))
sess.close

with tf.Session() as sess:
    output = sess.run(res)
    print(output)


#Log para datos de imagenes, las cuales son datos no distribuidos
config = tf.ConfigProto(log_device_placement = True)
config = tf.ConfigProto(allow_soft_placement = True)