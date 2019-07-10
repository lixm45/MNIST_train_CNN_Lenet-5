# 搭建LeNet-5 MNIST CNN网络

import matplotlib.pyplot as plt
import tensorflow as tf
from MNIST_import import mnist_read
# 读取数据
# 本次训练将不使用Validation集合
train_images, train_labels, validation_images, validation_labels, test_images, test_labels, tr, va, te = mnist_read()


# 设计超参数
# 定义输入
with tf.name_scope('inputs'):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="input")
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")

# minibatch训练的参数
sample_num = 55000 # 样本个数
epoch_num = 5 # 设置迭代次数
batch_size = 512 # 设置一个batch中包含样本的个数
batch_total = int(sample_num/batch_size) # 计算每一次迭代中包含的batch个数
# 网络超参数
keep_rate = tf.placeholder(tf.float32, name='keep_prob') # dropout超参数
lr = tf.placeholder(tf.float32, name='learn_rate') #学习率
L2_scale = 0.001
# 获取batch数据
images_batch, labels_batch = mini_batch_data(train_images, train_labels, batch_size=batch_size)
# 调整输入数据格式
images_batch_input = tf.reshape(images_batch, shape=(batch_size, 28, 28, 1))
test_images_input = tf.reshape(test_images, shape=(10000, 28, 28, 1))
# 搭建模型
# 卷积层
L1_cnn = tf.layers.conv2d(xs, filters=6, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu)
L1_ave = tf.layers.average_pooling2d(L1_cnn, pool_size=2, strides=2)
L1_dropout = tf.nn.dropout(L1_ave, keep_prob=keep_rate)

L2_cnn = tf.layers.conv2d(L1_dropout, filters=16, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu)
L2_ave = tf.layers.average_pooling2d(L2_cnn, pool_size=2, strides=2)
L2_dropout = tf.nn.dropout(L2_ave, keep_prob=keep_rate)

# 全连接层
L2_flatten = tf.layers.flatten(L2_dropout) #展平操作

L3_fc = tf.layers.dense(L2_flatten, units=120, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_scale))
L3_fc_dropout = tf.nn.dropout(L3_fc, keep_rate)

L4_fc = tf.layers.dense(L3_fc_dropout, units=80, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_scale))
L4_fc_dropout = tf.nn.dropout(L4_fc, keep_rate)

# logist 层
L5 = tf.layers.dense(L4_fc_dropout, units=10, kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_scale))

# 计算准确率
y_possibility = tf.nn.softmax(L5, axis=1)
y_predict_acc = tf.equal(tf.argmax(ys, axis=1), tf.argmax(y_possibility, axis=1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(y_predict_acc, tf.float32))

# 计算loss
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=L5)
    base_loss = tf.reduce_mean(cross_entropy)
    # 在loss中添加正则项
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = base_loss + tf.add_n(reg_loss)
# 设计优化器
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(lr).minimize(loss)
# 可视化
tf.summary.scalar('accuracy', accuracy)
# tf.summary.scalar('loss', loss)
summary_op = tf.summary.merge_all()

# 保存模型
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver.save(sess, "./model_CNN/model.ckpt")
    writer = tf.summary.FileWriter("logs_CNN/", sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    plt.ion()  # 开启interactive mode
    plt.figure(1)
    epoch_iter = [0]
    epoch_now = 0
    acc_now = 0
    acc_iter = [0]
    plt.clf()  # 清空画布上的所有内容
    try:
        for i in range(epoch_num):

            for j in range(batch_total):
                train_batch_data, train_batch_label = sess.run([images_batch_input, labels_batch])
                _, summary = sess.run([train, summary_op], feed_dict={xs: train_batch_data, ys:train_batch_label,
                                                                      keep_rate: 0.8, lr: 0.005})

                writer.add_summary(summary, i*j + j)
                # print('第{}次{}个batch迭代训练集准确率：{}'.format(i, j, sess.run(accuracy, feed_dict={xs: train_batch_data, ys: train_batch_label,
                #                                                                 keep_rate: 0.8, lr: 0.005})))

                epoch_now = i * batch_total + j
                epoch_iter.append(epoch_now)  # 模拟数据增量流入，保存历史数据
                acc_now = sess.run(accuracy, feed_dict={xs: train_batch_data, ys: train_batch_label,keep_rate: 0.8, lr: 0.008})
                acc_iter.append(acc_now)  # 模拟数据增量流入，保存历史数据
                plt.plot(epoch_iter, acc_iter, '-r')
                plt.pause(0.01)

        print('训练集准确率：{}'.format(acc_now))
        test_images_input_np = sess.run(test_images_input)
        print('测试集准确率：{}'.format(sess.run(accuracy, feed_dict={xs: test_images_input_np, ys: test_labels, keep_rate: 0.8, lr: 0.008})))

    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)
    plt.ioff()
    plt.show()



