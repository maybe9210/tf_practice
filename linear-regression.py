import tensorflow as tf
tf.set_random_seed(777)
# set_random_seed 함수 공부 필요 

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
# Variable 함수 공부 필요
# tensorflow 에서 자체적으로 만들어 내는 값.

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# 최적화 값을 찾는 코드

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run(
            [train, cost, W, b], feed_dict={X:[1,2,3], Y:[1,2,3]}
        )

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
    
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.3, 8.9]}))