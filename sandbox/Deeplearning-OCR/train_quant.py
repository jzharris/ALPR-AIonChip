#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import os.path as ops
import time
import numpy as np
from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from global_configuration import config
logger = log_utils.init_logger()
tf.logging.set_verbosity(tf.logging.ERROR)

from utils import *


# In[2]:


# one method to do quantization
def strategy_1(v_numpy):
    NUM_OF_LEVEL = 256
    NUM_OF_LEVEL -= 1 # Remove one, save this spot for [0] - NEW
    v_numpy_shape = v_numpy.shape
    v_numpy_flatten = v_numpy.flatten()
    min_np = v_numpy.min()
    max_np = v_numpy.max()
    gap = (max_np - min_np) / (NUM_OF_LEVEL - 1)
    #levels = np.asarray([(min_np + x * gap) for x in range(NUM_OF_LEVEL)])
    levels = np.asarray([(min_np + x * gap) for x in range(NUM_OF_LEVEL)] + [0]) # add 0, to pass pruned weights - NEW
    for i in range(len(v_numpy_flatten)):
        v_numpy_flatten[i] = min(levels, key=lambda x : abs(x-v_numpy_flatten[i]))
    v_numpy = v_numpy_flatten.reshape(v_numpy_shape)
    return v_numpy


def quantize_one_variable(sess, v):
    v_numpy = sess.run(v)
    new_v = strategy_1(v_numpy)
    assign_op = v.assign(new_v)
    return assign_op

default_ignore_quant_condition = lambda layer: 'W:0' not in layer.name and 'kernel:0' not in layer.name and 'w:0' not in layer.name

def quantize(sess, pruned_saver, checkpoint_save_path, iter_num,
             ignore_prune_condition = default_ignore_quant_condition):
    print("-----Starting Quant...------")
    update_operation = []
    for layer in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if ignore_prune_condition(layer):
            print("Ignoring ",layer.name+"...")
            continue
        print("Quantizing ", layer.name+"...")
        v = quantize_one_variable(sess, layer)
        update_operation.append(v)
    sess.run(update_operation)
    print("-----Finished Quant...------")
    #pruned_saver.save(sess, checkpoint_save_path, global_step=int(iter_num))

def gradient_stop(sess, opt, loss, global_step, ignore_prune_condition=default_ignore_quant_condition, verbose=True):
    pruned_train_gradient = []
    computed_losses = opt.compute_gradients(loss)
    print("Stopping gradients...")
    #print(tf.trainable_variables())
    for layer in tf.trainable_variables():
        with (tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))):
            layer_gradient = [x for x in computed_losses if layer.name==x[1].name][0]
            if ignore_prune_condition(layer):
                pruned_train_gradient.append(layer_gradient)
                continue
            layer_weights = sess.run(layer)

            mask = (layer_weights!=0) #(layer_weights >= threshold) | (layer_weights <= -threshold)

            # Report % pruned
            percent_pruned = 1-np.average(np.array(mask, dtype='int'))
            if (verbose):
                print("Percent Gradient Stopped: ", percent_pruned, "("+layer.name+")")

            pruned_train_gradient.append([tf.multiply(layer_gradient[0], mask), layer_gradient[1]])
        sess.run(tf.assign(layer,tf.multiply(layer,mask)))
    with (tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))):
        train_op = opt.apply_gradients(pruned_train_gradient, global_step=global_step)
    return train_op


# In[3]:


def train_shadownet_quant(dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param weights_path:
    :return:
    """
    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO().reader
    images, labels, imagenames = decoder.read_features(ops.join(dataset_dir, 'train_feature.tfrecords'),
                                                       num_epochs=None)
    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, imagenames], batch_size=32, capacity=1000+2*32, min_after_dequeue=100, num_threads=1)

    inputdata = tf.cast(x=inputdata, dtype=tf.float32)

    # initializa the net model
    shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=inputdata)

    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=25*np.ones(32)))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out, 25*np.ones(32), merge_repeated=False)

    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS, config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    opt=tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = opt.minimize(loss=cost, global_step=global_step)

    # Set saver configuration
    saver = tf.train.Saver(max_to_keep=0)
    model_save_dir = 'model/shadownet/checkpoints'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = model_save_dir + "/model.ckpt"
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    # Set the training parameters
    train_epochs = 100#500
    max_iters = 60
    display_steps = 10
    pruning_percent_per_iteration = 3

    with sess.as_default():
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for iteration in range(max_iters):
            print("Iteration %d/%d\n" % (iteration+1, max_iters))
            #saver.save(sess=sess, save_path=model_save_dir+"/model_pre_pruned.ckpt", global_step=iteration, write_meta_graph=False)
            #train_op = prune(sess, opt, cost, pruning_percent_per_iteration/100, global_step)
            train_op = gradient_stop(sess, opt, cost, global_step)
            quantize(sess, saver, os.path.join(model_save_dir, 'model_quantized.ckpt'), iteration)
            #print("Saved quantized model:",'model_quantized.ckpt-%d' % iteration)
             #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
             #    train_op = opt.minimize(loss=cost, global_step=global_step)
            #nonzero_weights = 0
            #total_weights = 0
            #for layer in tf.trainable_variables():
            #        layer_weights = sess.run(layer)
            #        #print(layer.name, np.count_nonzero(layer_weights)/np.prod(layer_weights.shape))
            #        nonzero_weights += np.count_nonzero(layer_weights)
            #        total_weights += np.prod(layer_weights.shape)
            #print("Reduced to %f percent of its original size." % (100*(nonzero_weights/total_weights)))
            saver.save(sess=sess, save_path=model_save_dir+"/model_post_quantization.ckpt", global_step=iteration, write_meta_graph=False)
            print("Saved at %s-%d" % (model_save_dir+"/model_post_quantization.ckpt", iteration))
            accs = test_shadownet(dataset_dir, model_save_dir+"/model_post_quantization.ckpt-"+str(iteration), verbose=False)
            print(accs)
            for epoch in range(train_epochs):
                _, c, seq_distance, preds, gt_labels = sess.run(
                    [train_op, cost, sequence_dist, decoded, input_labels])

                # calculate the precision
                preds = decoder.sparse_tensor_to_str(preds[0])
                gt_labels = decoder.sparse_tensor_to_str(gt_labels)

                accuracy = []

                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
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
                #
                if epoch % display_steps == 0:
                    logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, c, seq_distance, accuracy))
            saver.save(sess=sess, save_path=model_save_dir+"/model_post_training.ckpt", global_step=iteration, write_meta_graph=False)
            #saver.save(sess=sess, save_path=model_save_path, global_step=iteration, write_meta_graph=False)
            print("Saved at %s-%d" % (model_save_dir+"/model_post_training.ckpt", iteration))
            accs = test_shadownet(dataset_dir, model_save_dir+"/model_post_training.ckpt-"+str(iteration), verbose=False)
            print(accs)
            print("---------------------------------------------------")
        coord.request_stop()
        coord.join(threads=threads)
    tf.reset_default_graph()
    sess.close()

    return


# In[4]:


dataset_dir = "dataset"
#weights_path = "model/shadownet/shadownet_2019-02-14-03-49-38.ckpt-6076"
#weights_path = "model/shadownet/shadownet_2019-03-15-14-12-41.ckpt-46775"
#weights_path = "model/shadownet/checkpoints/working/best"
weights_path = "model/shadownet/checkpoints/working_pruning2/best.ckpt"
if not ops.exists(dataset_dir):
    raise ValueError('{:s} doesn\'t exist'.format(dataset_dir))
train_shadownet_quant(dataset_dir, weights_path)


# ## Trainable Layers:
# 
# 
# [<tf.Variable 'shadow/conv1/W:0' shape=(3, 3, 3, 64) dtype=float32_ref>, <tf.Variable 'shadow/conv2/W:0' shape=(3, 3, 64, 128) dtype=float32_ref>, <tf.Variable 'shadow/conv3/W:0' shape=(3, 3, 128, 256) dtype=float32_ref>, <tf.Variable 'shadow/conv4/W:0' shape=(3, 3, 256, 256) dtype=float32_ref>, <tf.Variable 'shadow/conv5/W:0' shape=(3, 3, 256, 512) dtype=float32_ref>, <tf.Variable 'shadow/BatchNorm/beta:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'shadow/BatchNorm/gamma:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'shadow/conv6/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'shadow/BatchNorm_1/beta:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'shadow/BatchNorm_1/gamma:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'shadow/conv7/W:0' shape=(2, 2, 512, 512) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/stack_bidirectional_rnn/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel:0' shape=(768, 1024) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/stack_bidirectional_rnn/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel:0' shape=(768, 1024) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel:0' shape=(768, 1024) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel:0' shape=(768, 1024) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'shadow/LSTMLayers/w:0' shape=(512, 37) dtype=float32_ref>]
