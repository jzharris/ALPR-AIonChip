#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-25 下午3:56
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : test_shadownet.py
# @IDE: PyCharm Community Edition
"""
Test shadow net script
"""
import os.path as ops
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math

import sys
sys.path.insert(0,'.')
from local_utils import data_utils
from crnn_model import crnn_model_relu
# from crnn_model import crnn_model as crnn_model_relu      # <-- uncomment to add drop activation to model
from global_configuration import config


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the test tfrecords data')
    parser.add_argument('--weights_path', type=str, help='Where you store the shadow net weights')
    parser.add_argument('--is_recursive', type=bool, help='If need to recursively test the dataset')

    return parser.parse_args()


def test_shadownet(dataset_dir, weights_path, is_vis=False, is_recursive=True):
    """

    :param dataset_dir:
    :param weights_path:
    :param is_vis:
    :param is_recursive:
    :return:
    """
    # Initialize the record decoder
    decoder = data_utils.TextFeatureIO().reader
    images_t, labels_t, imagenames_t = decoder.read_features(
        ops.join(dataset_dir, 'test_feature.tfrecords'), num_epochs=None)
    if not is_recursive:
        images_sh, labels_sh, imagenames_sh = tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                                                     batch_size=32, capacity=1000+32*2,
                                                                     min_after_dequeue=2, num_threads=4)
    else:
        images_sh, labels_sh, imagenames_sh = tf.train.batch(tensors=[images_t, labels_t, imagenames_t],
                                                             batch_size=32, capacity=1000 + 32 * 2, num_threads=4)

    images_sh = tf.cast(x=images_sh, dtype=tf.float32)

    # build shadownet
    net = crnn_model_relu.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=images_sh)

    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(32), merge_repeated=False)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    test_sample_count = 0
    for record in tf.python_io.tf_record_iterator(ops.join(dataset_dir, 'test_feature.tfrecords')):
        test_sample_count += 1
    loops_nums = int(math.ceil(test_sample_count / 32))
    # loops_nums = 100

    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Start predicting ......')
        if not is_recursive:
            predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
            imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
            imagenames = [tmp.decode('utf-8') for tmp in imagenames]
            preds_res = decoder.sparse_tensor_to_str(predictions[0])
            gt_res = decoder.sparse_tensor_to_str(labels)

            accuracy = []

            for index, gt_label in enumerate(gt_res):
                pred = preds_res[index]
                totol_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / totol_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Mean test accuracy is {:5f}'.format(accuracy))

            for index, image in enumerate(images):
                print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                    imagenames[index], gt_res[index], preds_res[index]))
                if is_vis:
                    plt.imshow(image[:, :, (2, 1, 0)])
                    plt.show()
        else:
            accuracy = []
            zero_acc = []
            one_acc = []
            two_acc = []
            more_acc = []
            for epoch in range(loops_nums):
                predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
                imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
                imagenames = [tmp.decode('utf-8') for tmp in imagenames]
                preds_res = decoder.sparse_tensor_to_str(predictions[0])
                gt_res = decoder.sparse_tensor_to_str(labels)

                for index, gt_label in enumerate(gt_res):
                    pred = preds_res[index]
                    totol_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if tmp == pred[i]:
                                correct_count += 1

                        # get other accuracies
                        z_count = 0
                        len_diff = len(gt_label) - len(pred)
                        if len_diff == 0:
                            # labels line up, simply count how each one fairs:
                            for i, tmp in enumerate(gt_label):
                                if tmp == pred[i]:
                                    z_count += 1
                        elif len_diff > 0:
                            # there are more chars in gt_label than pred, slide pred over gt_label
                            for j in range(len_diff + 1):
                                tmp_count = 0
                                for i, tmp in enumerate(pred):
                                    if tmp == gt_label[i+j]: # sliding window
                                        tmp_count += 1
                                if tmp_count > z_count:
                                    z_count = tmp_count
                        else:
                            # print(">>> {} vs {}".format(gt_label, pred))
                            # print(">>> len_diff: {}".format(len_diff))
                            # there are more chars in pred than gt_label, slide gt_label over pred
                            for j in range(-len_diff + 1):
                                tmp_count = 0
                                for i, tmp in enumerate(gt_label):
                                    if tmp == pred[i + j]:  # sliding window
                                        tmp_count += 1
                                # print(">>> {} ?= {} == {} correct".format(gt_label, pred[j:j+len(gt_label)], tmp_count))
                                if tmp_count > z_count:
                                    z_count = tmp_count
                        # z_count counts how many characters are correct, taking shifts into account
                        off_count = len(pred) - z_count # get how many chars the prediction is off by
                        print("{} vs {} = {} off".format(gt_label, pred, off_count))
                        zero_acc.append(0 if off_count == 0 else 1)
                        one_acc.append(0 if off_count == 1 else 1)
                        two_acc.append(0 if off_count == 2 else 1)
                        more_acc.append(0 if off_count >= 3 else 1)

                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / totol_count)
                        except ZeroDivisionError:
                            if len(pred) == 0: # null hypothesis
                                accuracy.append(1)
                            else:
                                accuracy.append(0)

                for index, image in enumerate(images):
                    print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                        imagenames[index], gt_res[index], preds_res[index]))
                    # if is_vis:
                    #     plt.imshow(image[:, :, (2, 1, 0)])
                    #     plt.show()

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            zero_acc = np.mean(np.array(zero_acc).astype(np.float32), axis=0)
            one_acc = np.mean(np.array(one_acc).astype(np.float32), axis=0)
            two_acc = np.mean(np.array(two_acc).astype(np.float32), axis=0)
            more_acc = np.mean(np.array(more_acc).astype(np.float32), axis=0)
            print('Overall acc:     {:5f}%'.format(accuracy*100))
            print('>=3 wrong acc:   {:5f}%'.format(more_acc*100))
            print('Two wrong acc:   {:5f}%'.format(two_acc*100))
            print('One wrong acc:   {:5f}%'.format(one_acc*100))
            print('Zero wrong acc:  {:5f}%'.format(zero_acc*100))

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test shadow net
    test_shadownet(args.dataset_dir, args.weights_path, args.is_recursive)
