import tensorflow as tf
import numpy as np
import WhatWhereAutoencoder as wwa
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class DeCNN_Vgg16:
    """
    A trainable version the DeCNN of Vgg16.
    """

    def __init__(self, vgg16_npy_path=None, trainable=True, dropout=1):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def reconstruct(self, feature_batch, train_mode=None):
        assert feature_batch.shape[1] == 1000

        # full connected layer, the reverse of the fc8 layer in vgg16

        self.fc8 = self.fc_layer(feature_batch, 1000, 4096, "fc8")
        self.relu8 = self.leaky_relu(self.fc8)
        if train_mode is not None:
            self.relu8 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu8, self.dropout), lambda: self.relu8)
        elif self.trainable:
            self.relu8 = tf.nn.dropout(self.relu8, self.dropout)

        # full connected layer, the reverse of the fc7 layer in vgg16

        self.fc7 = self.fc_layer(self.relu8, 4096, 4096, "fc7")
        self.relu7 = self.leaky_relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)


        # full connected layer, the reverse of the fc6 layer in vgg16

        self.fc6 = self.fc_layer(self.relu7, 4096, 25088, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = self.leaky_relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.reshape1 = tf.reshape(self.relu6, shape=[tf.shape(feature_batch)[0], 7, 7, 512])

        # an up-sampling layer followed with three convolution layer
        # the reverse of the conv5 layer in vgg16
        # batch*7*7*512    -->    batch*14*14*512

        self.upsample5 = wwa.upsample(self.reshape1, 2)
        self.conv5_1 = self.conv_layer(self.upsample5, 512, 512, "deconv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "deconv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "deconv5_3")

        # the reverse of the conv4 layer in vgg16
        # batch*14*14*512    -->    batch*28*28*256
        self.upsample4 = wwa.upsample(self.conv5_3, 2)
        self.conv4_1 = self.conv_layer(self.upsample4, 512, 512, "deconv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "deconv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 256, "deconv4_3")

        # the reverse of the conv3 layer in vgg16
        # batch*28*28*256    -->    batch*56*56*128
        self.upsample3 = wwa.upsample(self.conv4_3, 2)
        self.conv3_1 = self.conv_layer(self.upsample3, 256, 256, "deconv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "deconv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 128, "deconv3_3")

        # the reverse of the conv2 layer in vgg16
        # batch*56*56*128    -->    batch*112*112*64

        self.upsample2 = wwa.upsample(self.conv3_3, 2)
        self.conv2_1 = self.conv_layer(self.upsample2, 128, 128, "deconv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 64, "deconv2_2")

        # the reverse of the conv1 layer in vgg16
        # batch*112*112*64    -->    batch*224*224*3

        self.upsample1 = wwa.upsample(self.conv2_2, 2)
        self.conv1_1 = self.conv_layer(self.upsample1, 64, 64, "deconv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 3, "deconv1_2")

        print("upsamlpe5 shape  {}".format(self.upsample5.shape))
        print("upsamlpe4 shape  {}".format(self.upsample4.shape))
        print("upsamlpe3 shape  {}".format(self.upsample3.shape))
        print("upsamlpe2 shape  {}".format(self.upsample2.shape))
        print("upsamlpe1 shape  {}".format(self.upsample1.shape))

        self.data_dict = None

        return 0

    def leaky_relu(self, x, leak = 0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5*(1+leak)
            f2 = 0.5*(1-leak)
            return f1*x + f2*tf.abs(x)


    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
