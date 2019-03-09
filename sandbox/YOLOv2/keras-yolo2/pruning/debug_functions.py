import tensorflow as tf
import os

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = './dataset/cifar/'
FIGS_DIR = 'figs'


def plot_weight_dist(sess, bins=100, grad_mask_consts=None, title='', fig_name=None, verbose=False, zero_mask=False,
                     is_pruning=True, figs_dir='figs/', prune_threshold=0):
    print('Plotting weight distribution...')

    if not os.path.isdir(figs_dir):
        os.mkdir(figs_dir)

    if is_pruning:
        if not os.path.isdir('{}/p{}'.format(figs_dir, prune_threshold)):
            os.mkdir('{}/p{}'.format(figs_dir, prune_threshold))
        fig_dir = '{}/p{}'.format(figs_dir, prune_threshold)
    else:
        if not os.path.isdir('{}/q100'.format(figs_dir)):
            os.mkdir('{}/q100'.format(figs_dir))
        fig_dir = '{}/q100'.format(figs_dir)

    # evaluate variables (get weights)
    vars = tf.trainable_variables(scope='fcn')
    vars_vals = sess.run(vars)
    all_vals = []

    if grad_mask_consts is not None:
        for var, val in zip(vars, vars_vals):
            mask_val = sess.run(grad_mask_consts[var.name])
            val_np = np.array(val).flatten()
            mask_np = np.array(mask_val).flatten()
            for idx in range(len(val_np)):
                if mask_np[idx] != 0:
                    all_vals.append(val_np[idx])
    elif zero_mask:
        for var, val in zip(vars, vars_vals):
            val_np = np.array(val)
            for v in val_np.flatten():
                if v != 0:
                    all_vals.append(v)
    else:
        for var, val in zip(vars, vars_vals):
            val_np = np.array(val)
            for v in val_np.flatten():
                all_vals.append(v)

    all_vals = np.array(all_vals)
    if verbose:
        print('min:', all_vals.min(), '- max:', all_vals.max())

    plt.title(title)
    plt.hist(all_vals, bins=bins, range=(-1, 1))

    if fig_name is not None:
        plt.savefig('{}/{}'.format(fig_dir, fig_name))
        plt.close()
    else:
        plt.show()


def print_inference(sess, testSet, images, fine_labels, acc_op, top_5_accuracy_op, loss_op, batch_size):
    test_batches = [{'data': testSet['data'][i:i + batch_size],
                     'labels': testSet['labels'][i:i + batch_size]
                     } for i in range(0, len(testSet['data']), batch_size)]

    cum_acc = 0
    cum_top_5 = 0
    cum_loss = 0

    for test_batch in test_batches:
        feed_dict = {
            images: test_batch['data'],
            fine_labels: test_batch['labels'],
        }
        inference_accuracy, top_5_accuracy, inference_loss = sess.run([acc_op, top_5_accuracy_op, loss_op], feed_dict=feed_dict)
        cum_acc += inference_accuracy
        cum_top_5 += top_5_accuracy
        cum_loss += inference_loss

    return cum_acc / len(test_batches), cum_top_5 / len(test_batches), cum_loss / len(test_batches)