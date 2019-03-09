#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : demo_shadownet.py
# @IDE: PyCharm Community Edition
"""
Use shadow net to recognize the scene text
"""
import tensorflow as tf
import os
import os.path as ops
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import csv
#import os
import sys
import numpy as np
sys.path.append(r'C:\Users\vinayak\Documents\github\CRNN_Tensorflow')
try:
    from cv2 import cv2
except ImportError:
    pass

from crnn_model import crnn_model
from global_configuration import config
from local_utils import log_utils, data_utils
from tesseract_recog import recognize_img

logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Where you store the image',
                        default='../data/test_images/test_02.jpg')
    parser.add_argument('--weights_path', type=str, help='Where you store the weights',
                        default='../model/shadownet/shadownet_2017-10-17-11-47-46.ckpt-199999')

    return parser.parse_args()





def recognize(image_path, weights_path, is_vis=False):
    """

    :param image_path:
    :param weights_path:
    :param is_vis:
    :return:
    """
    tf.reset_default_graph()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 32))
    image = np.expand_dims(image, axis=0).astype(np.float32)

    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')

    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow'):
        net_out, score = net.build_shadownet(inputdata=inputdata)
        dime = tf.reduce_max(tf.nn.softmax(net_out), reduction_indices=[2]) 
    #score = tf.nn.softmax(net_out)
    decodes, log = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)


    decoder = data_utils.TextFeatureIO()

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        net_out, preds, softmax_value,log_value = sess.run([net_out,decodes,dime,log], feed_dict={inputdata: image})
        #print(preds)
        
        
        #print("softmax score:",np.mean(np.max(softmax_value,axis=2)))
        preds = decoder.writer.sparse_tensor_to_str(preds[0])
        print(preds[0],np.min(softmax_value),np.exp(log_value) )
        
        #logger.info('Predict image {:s} label {:s}'.format(ops.split(image_path)[1], preds[0]))

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
            plt.show()

        sess.close()

    return preds,log_value


if __name__ == '__main__':
    # Inti args
    args = init_args()
    #if not ops.exists(args.image_path):
        #raise ValueError('{:s} doesn\'t exist'.format(args.image_path))
        
    file_path = 'test.csv'
    fieldnames = ['image','label','score']
    with open('test.csv', 'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for file in os.listdir("../data/test_images_1"):
            print("processing:"+str(file))
            # recognize the image
            predicted, score = recognize(image_path="../data/test_images_1/"+file, weights_path=args.weights_path)
            # predicted, score = recognize_img("data/test_images/"+file)
            #writer.writerow({'image':str(file), 'label': str(predicted),'score': str(score)})
    
