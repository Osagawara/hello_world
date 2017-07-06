import numpy as np
import tensorflow as tf
import sys
import os
import vgg16
import deCNN_vgg16
import batch_features

WEIGHT_DECAY_FACTOR = 0.0004
EPOCHS = 40
imlist = []
data_dir = '/tmp/tensorflow/mnist/input_data'
log_dir = '/tmp/tensorflow/mnist/logs/mnist_with_summaries'


with tf.Session() as sess:
    images = tf.placeholder("float", [None, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.build(images)

    # with tf.variable_scope("decnn", regularizer=tf.nn.l2_loss()):

    decnn = deCNN_vgg16.DeCNN_Vgg16()
    decnn.reconstruct(vgg.relu6)
    weights_norm = tf.reduce_sum(
        input_tensor=WEIGHT_DECAY_FACTOR*tf.pack(
            [tf.nn.l2_loss(i) for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        ),
        name='weight_norm'
    )

    recon_loss = tf.nn.l2_loss(images*255.0 - decnn.conv1_2, name='recon_loss')
    total_loss = tf.add(recon_loss, weights_norm, 'total_loss')

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               5000, 0.96, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('recon_loss', recon_loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    tf.global_variables_initializer().run()
    path = '/home/kakeru/Downloads/n01440764'
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.JPEG')]
    for i in range(EPOCHS):
        batch = batch_features.image_batch(imlist[i*32, (i+1)*32])
        feed_dict = {images:batch}
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        train_writer.add_summary(summary, i)

    decnn.save_npy(sess)
    train_writer.close()
    test_writer.close()

