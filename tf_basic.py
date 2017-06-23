import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1, x2)

# the better way to do it so that we don't have to close the session afterwards
with tf.Session() as sess:
    output = sess.run(result)
    print(output)
    
# we can access the variables here but we cannot access the session (which has already been closed)
print(output)