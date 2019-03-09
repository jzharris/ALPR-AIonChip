from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
import cv2
from datetime import datetime

# from model_only import fcn
# from debug_functions import plot_weight_dist, print_inference
from prune_network import prune_layers, check_pruned_weights, print_pruned_weights
# from tqdm import tqdm

import keras

# NUM_CLASS = 10
# MAX_STEPS = [30000, 30000, 30000, 30000, 30000, 30000]
# LOG_FREQUENCY = None
# BATCH_SIZE = 64
# LEARNING_RATE = 0.001
# MODEL_SAVING_FREQUENCY = None
# TRAIN_DIR = './checkpoints/train/'
# LOGS_DIR = './tensorboard/logs/'
# white_list = []# ['fcn/conv5/bias:0', 'fcn/norm5/gamma:0', 'fcn/norm5/beta:0']
# white_regex = []# ['bias', 'gamma', 'beta']
# skip_first = True     # if True, will skip the first cycle of training and prune the network before training begins.


def main():
    # grad mask dict placeholder
    grad_mask_consts = None

    # # load data
    # print("Loading datasets...")
    # trainSet, testSet, labelNames = data_loader('./dataset/cifar/')
    # print("Dataset loading complete.")
    #

    # reset default graph
    tf.reset_default_graph()

    #
    # global_step = tf.train.get_or_create_global_step()
    #
    # # define optimizer
    # opt = tf.train.GradientDescentOptimizer(
    #     learning_rate=LEARNING_RATE,
    #     # beta1=0.9,
    #     # beta2=0.999,
    #     # epsilon=1e-08,
    #     use_locking=False,
    #     name='GD'
    # )
    #
    # images = tf.placeholder(name='images', dtype=tf.float32, shape=[None, 32, 32, 3])
    # fine_labels = tf.placeholder(name='fine_labels', dtype=tf.int32, shape=[None])
    #
    # logits = fcn(images, is_training=True)
    # # probs = tf.nn.softmax(logits)
    # # pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    #
    # # loss metrics:
    # loss = tf.losses.softmax_cross_entropy(tf.one_hot(fine_labels, NUM_CLASS), logits)
    # training_loss = tf.summary.scalar("training_loss", loss)
    # validation_loss = tf.summary.scalar("validation_loss", loss)
    #
    # # accuracy metrics:
    # top_1 = tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), fine_labels), tf.float32)
    # accuracy = tf.reduce_mean(top_1)
    # training_accuracy = tf.summary.scalar("training_accuracy", accuracy)
    # validation_accuracy = tf.summary.scalar("validation_accuracy", accuracy)
    #
    # top_5 = tf.reduce_sum(tf.cast(tf.equal(tf.nn.top_k(logits, k=5, sorted=True)[1],
    #                                        tf.expand_dims(fine_labels, 1)), tf.float32), axis=1)
    # top_5_accuracy = tf.reduce_mean(top_5)
    #
    # # init summary writer:
    # writer = tf.summary.FileWriter(LOGS_DIR)

    # create graph from pretrained h5 (Keras) file
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    model = keras.models.load_model("../lp_seg_mobilenet.h5")
    sess = keras.backend.get_session()
    # save_path = saver.save(sess, "../model.ckpt")

    step_ = 0
    for it in range(len(MAX_STEPS)):

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if grad_mask_consts is not None:
                # Get all trainable parameters
                vars = tf.trainable_variables(scope='fcn')

                # Compute the gradients for a list of variables.
                train_grads = opt.compute_gradients(loss, vars)

                # Apply mask. orig_grads_and_vars is a list of tuples (gradient, variable).
                pruned_train_gradient = [
                    (tf.multiply(tf.cast(grad_mask_consts[gv[1].name], tf.float32), gv[0]), gv[1]) for gv in train_grads]

                # Ask the optimizer to apply the masked gradients.
                train_op = opt.apply_gradients(pruned_train_gradient, global_step=global_step)
            else:
                train_op = opt.minimize(loss, global_step=global_step)

        # training setups
        saver = tf.train.Saver(max_to_keep=100)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
            )
        )
        sess.run(init)

        # try to load pre-trained models
        if tf.train.get_checkpoint_state(TRAIN_DIR) is not None:
            restorer = tf.train.Saver()
            restorer.restore(sess, tf.train.latest_checkpoint(TRAIN_DIR))
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), TRAIN_DIR))

        ################################################################################################################
        # Train the network
        if (it == 0 and not skip_first) or it > 0:
            # main loop
            for step in range(MAX_STEPS[it]):
                ############################################################################################################
                start_time = time.time()

                ############################################################################################################
                # training set
                # prepare data
                train_batch = sample_batch(trainSet, BATCH_SIZE)
                # feed dict
                feed_dict = {
                    images: train_batch['data'],
                    fine_labels: train_batch['labels'],
                }
                _, loss_value, train_loss, accuracy_value, train_acc, global_step_value = sess.run([train_op,
                                                                                                    loss, training_loss,
                                                                                                    accuracy, training_accuracy,
                                                                                                    global_step],
                                                                                                   feed_dict=feed_dict)
                writer.add_summary(train_acc, step_)
                writer.add_summary(train_loss, step_)

                ############################################################################################################
                # test set
                # prepare data
                test_batch = sample_batch(testSet, BATCH_SIZE)
                # feed dict
                feed_dict = {
                    images: test_batch['data'],
                    fine_labels: test_batch['labels'],
                }
                valid_accuracy_value, valid_acc, valid_loss_value, valid_loss = sess.run([accuracy, validation_accuracy,
                                                                                          loss, validation_loss],
                                                                                         feed_dict=feed_dict)
                writer.add_summary(valid_acc, step_)
                writer.add_summary(valid_loss, step_)

                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                ############################################################################################################

                # log
                if (LOG_FREQUENCY is not None and step % LOG_FREQUENCY == 0) or (step + 1) == MAX_STEPS[it]:
                    num_examples_per_step = BATCH_SIZE
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration
                    format_str = (
                        '%s: step %d, examples %d, loss = %.9f accuracy = %.4f (%.3f examples/sec; %.3f sec/batch)'
                    )
                    print(
                        format_str % (
                            datetime.now(), step, BATCH_SIZE * step,
                            loss_value,
                            accuracy_value,
                            examples_per_sec, sec_per_batch
                        )
                    )

                # Save the model checkpoint periodically
                if (MODEL_SAVING_FREQUENCY is not None and step % MODEL_SAVING_FREQUENCY == 0) or (step + 1) == MAX_STEPS[it]:
                    checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=int(global_step_value))

                step_ += 1

        ################################################################################################################
        # Inference measurement:
        inf_accuracy, inf_top_5, inf_loss = print_inference(sess, testSet, images, fine_labels, accuracy, top_5_accuracy, loss, it)
        print('>>>\tInference top_1 acc BEFORE pruning, it_{}: {}'.format(it, inf_accuracy))
        print('>>>\tInference top_5 acc BEFORE pruning, it_{}: {}'.format(it, inf_top_5))

        ################################################################################################################
        # Pruning step:
        plot_weight_dist(sess, bins=200, title='Weight distribution before pruning, iteration {}'.format(it),
                         fig_name='before_it{}'.format(it), verbose=False, zero_mask=it > 0)

        grad_mask_consts, global_step_value = prune_layers(sess, global_step, grad_mask_consts, white_list, white_regex)
        check_pruned_weights(sess, grad_mask_consts, it)
        print_pruned_weights(sess, grad_mask_consts)

        plot_weight_dist(sess, bins=200, grad_mask_consts=grad_mask_consts,
                         title='Weight distribution after pruning, iteration {}'.format(it),
                         fig_name='after_it{}'.format(it), verbose=False)

        ################################################################################################################
        # Inference measurement:
        inf_accuracy, inf_top_5, inf_loss = print_inference(sess, testSet, images, fine_labels, accuracy, top_5_accuracy, loss, it)
        print('>>>\tInference top_1 acc AFTER pruning, it_{}: {}'.format(it, inf_accuracy))
        print('>>>\tInference top_5 acc AFTER pruning, it_{}: {}'.format(it, inf_top_5))

        # Save the model checkpoint periodically
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=int(global_step_value))


if __name__ == '__main__':
    main()
