# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import vgg16
import utils

import os
import re

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def batch_single_set(images):
    '''

    :param images: a list of the path of images
    :return batch: numpy array of image content with 3 chanels
            image_path: the path of selected images
    '''
    image_array = []
    image_path = []
    for i in images:
        try:
            a = utils.load_image(i)
            if len(a.shape) == 3:
                image_array.append(a.reshape(1, 224, 224, 3))
                image_path.append(i)
        except IOError:
            pass
        except TypeError:
            pass
        except ValueError:
            pass
        finally:
            pass

    batch = np.concatenate(image_array, 0)
    return batch, image_path


# create a list of images
# imageset_list is the file directories of different kinds of images


imageset_path = '/raid/workspace/wangjunjun/imagenet/ILSVRC2012_img_train/'
imageset_list = list(os.path.join(imageset_path, f) for f in os.listdir(imageset_path) if not f.endswith(('.tar', '.txt')) )

# tensorflow input batch shouldn't be too large, otherwise it will terminate with error

sub_batch = 50

with tf.device('/gpu:0'):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        images = tf.placeholder("float", [None, 224, 224, 3])

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        for s in imageset_list[10:20]:
            print(s)
            image_list = list(os.path.join(s, f) for f in os.listdir(s) if f.endswith('.JPEG'))
            selected_images = []
            features = []

            for i in range(int(len(image_list) / sub_batch)):
                batch, temp= batch_single_set(image_list[(sub_batch * i): (sub_batch * (i + 1))])
                selected_images.extend(temp)
                feed_dict = {images:batch}
                fc8 = sess.run(vgg.fc8, feed_dict=feed_dict)
                features.append(fc8)
                print("sub_batch {} OK".format(i))

            features = np.concatenate(features, 0)
            npy_path = "features/features_imagenet_" + re.split('/', s)[-1] + "_vgg16_fc8.npy"
            np.save(npy_path, features)
            f = open('images/selected_images_' + re.split('/', s)[-1] + ".txt",'w')
            for i in selected_images:
                f.write(i + '/n')

            f.close()
            print(npy_path + "  OK")



