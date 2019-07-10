from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

def mnist_read():
    # 下载MNIST数据集
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 读取数据集到
    # 训练集
    train_images = mnist.train.images # 返回的 train_images 是 (55000, 784) 多维数组
    train_labels = mnist.train.labels # (55000, 10), 数据标签已经对应为张量为10的矩阵

    # 验证集
    validation_images = mnist.validation.images #（5000,784）
    validation_labels = mnist.validation.labels

    # 测试集
    test_images = mnist.test.images #(10000, 784)
    test_labels = mnist.test.labels

    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels, \
           mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples


# train_images, train_labels, validation_images, validation_labels, test_images, test_labels, t, v, t = mnist_read()
# print(t, v, t)
# 显示图片
# plt.imshow(train_images[:, 1].reshape(28, 28), cmap='Greys_r')
# plt.show()



