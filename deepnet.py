import tensorflow as tf
# input data > weight > hidden layer 1 (activation function) > weights > hidden layer 2
# (activation function) > weights > output layer
# compare output with intended output > cost or loss function (cross entropy)
# optimization function (optimizer) > minimize cost (AdamOptimizer, SGD, AdaGrad)
# backpropagation
# feedforward + backpropagation = epoch
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
# 10 classes, 0-9
# init a deep-net with 3 hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# the utility of specifying a placeholder is that if misshaped datasets are sent in, tensorflow will throw an error
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network(data):
    # (input_data * weights) + biases
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([784, n_nodes_hl1], stddev=0.1)), 
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)), 
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)), 
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)), 
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_classes]))}
    
    # Having a small positive initial bias means the neurons are more likely to fire in average. 
    # That speeds up training because backpropagation relies on how much impact each neuron has on the output, 
    # and inactive neurons have no impact whatsoever. It doesn't really matter if they're all the same, 
    # since the bias just controls the activation threshold, so we set it to a constant.
    # Now, having a large amplitude for the weights makes the network more unstable, and thus harder to train, 
    # since each learning step may change the functional behavior a lot. So we start those values randomly (like sentdex did), 
    # but with a smaller standard deviation (0.1) and without any values larger than 0.2 in magnitude 
    # (which is what the truncated normal distribution does). That ensures we get a better initial condition, 
    # so training goes smoother.
    # There's some work out there about how to pick a good initialization, but people often just try a bunch of variations 
    # from what I suggested until they find what works best.

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # ReLURectified linear something. A low-cost activation function.
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),  hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output

def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # cycles of feedforward and backprop
    hm_epochs = 15
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, ' completed out of ', hm_epochs, 'loss: ', epoch_loss)
            
        # checks if the maximum heated-up vector in predition matches the label
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
train_neural_network(x)