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

#default_ignore_prune_condition = lambda layer: 'kernel:0' not in layer.name and 'w:0' not in layer.name and 'W:0' not in layer.name
#default_ignore_prune_condition = lambda layer: 'kernel' not in layer.name
default_ignore_prune_condition = lambda layer: 'W:0' not in layer.name
#default_ignore_prune_condition = lambda layer: 'W:0' not in layer.name

def prune(sess, opt, loss, percent_to_prune, global_step, ignore_prune_condition=default_ignore_prune_condition, verbose=True):
    pruned_train_gradient = []
    computed_losses = opt.compute_gradients(loss)
    print("Pruning...")
    #print(tf.trainable_variables())
    for layer in tf.trainable_variables():
        with (tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))):
            layer_gradient = [x for x in computed_losses if layer.name==x[1].name][0]
            if ignore_prune_condition(layer):
                layer_weights = sess.run(layer)
                mask = (layer_weights!=0)
                pruned_train_gradient.append([tf.multiply(layer_gradient[0], mask), layer_gradient[1]])
                #pruned_train_gradient.append(layer_gradient)
                continue
            layer_weights = sess.run(layer)

            # Calculate threshold and masks
            magnitudes = np.abs(layer_weights.flatten())
            magnitudes = magnitudes[magnitudes!=0] # Keep the given percent of the un-pruned weights
            magnitudes.sort()
            threshold = magnitudes[int(magnitudes.shape[0]*percent_to_prune)]
            mask = (layer_weights >= threshold) | (layer_weights <= -threshold)

            # Report % pruned
            percent_pruned = 1-np.average(np.array(mask, dtype='int'))
            if (verbose):
                print("Percent Pruned: ", percent_pruned, "("+layer.name+")")

            pruned_train_gradient.append([tf.multiply(layer_gradient[0], mask), layer_gradient[1]])
        sess.run(tf.assign(layer,tf.multiply(layer,mask)))
    with (tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))):
        train_op = opt.apply_gradients(pruned_train_gradient, global_step=global_step)
    return train_op

def train_shadownet(dataset_dir, weights_path=None, train_epochs=200, max_iters=60, pruning_percent_per_iteration=3, start_iter=0, display_steps=10, batch_size=32):
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
        tensors=[images, labels, imagenames], batch_size=batch_size, capacity=1000+2*batch_size, min_after_dequeue=100, num_threads=1)

    inputdata = tf.cast(x=inputdata, dtype=tf.float32)

    # initializa the net model
    shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=inputdata)

    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=25*np.ones(batch_size)))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out, 25*np.ones(batch_size), merge_repeated=False)

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
    #model_save_path = model_save_dir + "/model.ckpt"
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    # Set the training parameters
    #train_epochs = 200#500
    #max_iters = 60
    #display_steps = 10
    #pruning_percent_per_iteration = 3

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

        for iteration in range(start_iter, max_iters):
            print("Iteration %d/%d\n" % (iteration+1, max_iters))
            #saver.save(sess=sess, save_path=model_save_dir+"/model_pre_pruned.ckpt", global_step=iteration, write_meta_graph=False)
            train_op = prune(sess, opt, cost, pruning_percent_per_iteration/100, global_step)
            #nonzero_weights = 0
            #total_weights = 0
            #for layer in tf.trainable_variables():
            #        layer_weights = sess.run(layer)
            #        #print(layer.name, np.count_nonzero(layer_weights)/np.prod(layer_weights.shape))
            #        nonzero_weights += np.count_nonzero(layer_weights)
            #        total_weights += np.prod(layer_weights.shape)
            #print("Reduced to %f percent of its original size." % (100*(nonzero_weights/total_weights)))
            saver.save(sess=sess, save_path=model_save_dir+"/model_post_pruning.ckpt", global_step=iteration, write_meta_graph=False)
            print("Saved at %s-%d" % (model_save_dir+"/model_post_pruning.ckpt", iteration))
            accs = test_shadownet(dataset_dir, model_save_dir+"/model_post_pruning.ckpt-"+str(iteration), verbose=False)
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

dataset_dir = "dataset"
#start_iter=0
#weights_path='model/shadownet/checkpoints/working_quant/best.ckpt'
#weights_path = "model/shadownet/shadownet_2019-03-15-14-12-41.ckpt-46775"
weights_path = "model/shadownet/checkpoints/final_checkpoints/ckpt1/best.ckpt"
start_iter = 17
# weights_path = "model/shadownet/checkpoints/model_post_training.ckpt-%d" % (start_iter-1)
if not ops.exists(dataset_dir):
    raise ValueError('{:s} doesn\'t exist'.format(dataset_dir))
if not ops.exists(weights_path+".index"):
    raise ValueError('{:s} doesn\'t exist'.format(weights_path))

train_epochs=2000
max_iters=60
pruning_percent=4
display_steps=50
batch_size=512

train_shadownet(dataset_dir, weights_path, train_epochs=train_epochs, max_iters=max_iters, pruning_percent_per_iteration=pruning_percent, start_iter=start_iter, display_steps=display_steps, batch_size=batch_size)
