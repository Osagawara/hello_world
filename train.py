import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import warnings
warnings.filterwarnings("ignore")

import math
import deCNN_vgg16
from test_vgg16_img import batch_single_set

batch_size = 20
epoch = 20
keep_prob = 0.9
global_step = 0

log_dir = './logs'
feature_dir = '../vgg/features'
image_dir = '../vgg/images'

feature_list = list(os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.npy') )
image_list = list(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.txt') )

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    features_holder = tf.placeholder(tf.float32, [batch_size, 1000])
    train_mode = tf.placeholder(tf.bool)
    images_holder = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

    deCNN = deCNN_vgg16.DeCNN_Vgg16(dropout=keep_prob)
    deCNN.reconstruct(features_holder, train_mode)

    cost = tf.reduce_sum((deCNN.conv1_2 - images_holder) ** 2)
    tf.summary.scalar(name='loss', tensor=cost)
    learning_rate = tf.train.exponential_decay(0.001, 100000, 1000, 0.98)
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        for j in range(len(feature_list)):
            features = np.load(feature_list[j])
            f = open(image_list[j])
            images = list(s.strip() for s in f.readlines())
            f.close()

            for k in range(int(math.ceil(len(feature_list) / batch_size))):
                feature_batch = features[k*batch_size : (k+1)*batch_size]
                image_batch, _= batch_single_set(images[k*batch_size : (k+1)*batch_size])
                data_dict = {features_holder:feature_batch,
                             train_mode:True,
                             images_holder:image_batch}

                summary, _ = sess.run([merged, train], feed_dict=data_dict)

                train_writer.add_summary(summary, global_step)
                global_step += 1

        deCNN.save_npy(sess=sess)

    train_writer.close()


