import tensorflow as tf
import matplotlib.pyplot as plt
# 그래프 그려지는 라이브러리

# X = [1,2,3]
# Y = [1,2,3]
# W = tf.placeholder(tf.float32)
x_data = [1,2,3]
y_data = [1,2,3]
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis-Y))

learing_rate = 0.1
gradient = tf.reduce_mean(( W * X - Y ) * X)
descent = W - learing_rate * gradient
update = W.assign(descent)
# 최적화 된 W를 찾는 코드들...

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
# 굳이 미분하지 않아도, 해당 코드를 통해 자동으로 최적값 W를 찾을 수 있다.

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# W_val = []
# cost_val = []

# for i in range(-30, 50):
#     feed_W = i * 0.1
#     curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
#     W_val.append(curr_W)
#     cost_val.append(curr_cost)
    
# plt.plot(W_val, cost_val)
# plt.show()

for step in range(21):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
