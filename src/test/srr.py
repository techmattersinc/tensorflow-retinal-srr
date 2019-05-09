import numpy as np
import scipy.misc
import cv2
import dlib
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../../utils')
sys.path.append('../../vgg19')
from srgan import SRGAN
import glob
import os

x = tf.placeholder(tf.float32, [None, 24, 24, 3])
is_training = tf.placeholder(tf.bool, [])

model = SRGAN(x, is_training, 16)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, '../backup/latest')

test_dir = '/Users/mayank/work/techmatters.ai/devpost_challenge/data/raw_input_video/mayank_left_04_29/retina_raw/*png'

def plot_img(img, filename):
    im = np.uint8((img+1)*127.5)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (96,96))
    print('finished:' + filename)
    scipy.misc.imsave('srr_results/' +  filename , im)

for file in glob.glob(test_dir):
    #file = '../../../../devpost_challenge/data/raw_input_video/mayank_left_04_29/retina_raw/out23.png'
    print(file)
    img = cv2.imread(file)
    filename = os.path.basename(file)

    out_ = np.zeros((96*4,96*4,3))

    h, w = img.shape[:2]
    img = img / 127.5 - 1
    for row in range(4):
        for col in range(4):
            input_ = np.zeros((16, 24, 24, 3))
            input_[0] = img[24*row:24*row+24, 24*col:24*col+24, :]
    
            mos, fake = sess.run( [model.downscaled, model.imitation], feed_dict={x: input_, is_training: False})
            #print( fake[0].shape )
            out_[96*row:96*row+96,96*col:96*col+96,:] = fake[0]
            
    plot_img(out_, filename)



