"""Adam for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np

DATA_DIR = './dataset/cifar/'
FIGS_DIR = 'figs'


def prune_layers(sess, prune_threshold, grad_mask_consts_old=None, white_list=None, white_regex=None):
    print('Pruning parameters one layer at a time...')

    if prune_threshold > 1 or prune_threshold < 0:
        raise Exception("Invalid prune threshold. Must be between 0 and 1")

    # prepare a gradient mask for each layer:
    grad_mask_consts_new = {}

    # evaluate variables (get weights)
    vars = tf.trainable_variables()
    vars_vals = sess.run(vars)

    # set everything under threshold to zero
    for idx, (var, val) in enumerate(zip(vars, vars_vals)):
        print('>>> {}'.format(var.name))

        # create np array of weights
        val_np = np.array(val)

        # exclude the already-pruned weights from the % calculation
        flattened = val_np.flatten()
        n = len(flattened)
        if grad_mask_consts_old is not None and var.name in grad_mask_consts_old.keys():
            mask = np.array(sess.run(grad_mask_consts_old[var.name]))
            already_masked = len(mask[mask == 0].flatten())
        else:
            already_masked = 0

        # mask out XX% of the unpruned weights (pruned weights measured by already_masked)
        outliers = int(np.round(0.5 * (1 - prune_threshold) * (n - already_masked)))

        # create the mask
        layer_weights = np.ones(val_np.shape)
        if (white_list is not None and var.name in white_list):
            print(">>>\t not pruning '{}', it is part of the whitelist".format(var.name))
        else:
            skip = False
            if white_regex is not None:
                for white in white_regex:
                    if white in var.name:
                        skip = True
                        break

            if skip:
                print(">>>\t not pruning '{}', it is part of the whitelist".format(var.name))
            else:
                sorted_full = np.dstack(np.unravel_index(np.argsort(flattened), val_np.shape))[0]
                sorted_full_prune = sorted_full[outliers : n - outliers]
                for prune in sorted_full_prune:
                    layer_weights[tuple(prune)] = 0

        # apply mask to original weights
        for l, x in zip(np.nditer(layer_weights, op_flags=['readwrite']), np.nditer(val_np.ravel(), op_flags=['readwrite'])):
            x[...] = l * x

        # save the zeroed weights to the session
        sess.run([tf.assign(var, val_np)])

        # add the mask to the list of grad_mask_consts:
        grad_mask_consts_new[var.name] = tf.constant(layer_weights)

    # # return the gradient mask for later use
    # global_step_value = sess.run(global_step)

    return grad_mask_consts_new# , global_step_value


def check_pruned_weights(sess, grad_mask_consts, prune_threshold, it):
    print('Checking that the pruned weights were not modified...')

    leaked_pruned_weights = 0
    total = 0
    original = 0

    # evaluate variables (get weights)
    vars = tf.trainable_variables()
    vals = sess.run(vars)

    # print([v.name for v in vars])

    # set everything under threshold to zero
    for idx, (var, val) in enumerate(zip(vars, vals)):
        # create np array of weights
        val_np = np.array(val)

        # mask out the un-pruned weights
        mask = np.array(sess.run(grad_mask_consts[var.name]))
        val_masked = val_np[mask == 0]

        # count how many of these are != 0
        leaked_pruned_weights += np.count_nonzero(val_masked)

        # count total number of pruned weights
        total += len(val_masked.flatten())

        # count total number of weights in this layer
        original += len(val_np.flatten())

    total_pruned = total
    count = leaked_pruned_weights

    percentage = total_pruned / original * 100
    true_percentage = (1 - (1-prune_threshold)**(it+1)) * 100

    print('>>>\t{} of the {} weights that should have been pruned are NONzero (should be 0)'.
          format(count, total_pruned, percentage))
    print('>>>\t{} of the {} total weights have been pruned ({:.6f}% of original, should be {:.6f}%)'.
          format(total_pruned, original, percentage, true_percentage))
    return count == 0


def print_pruned_weights(sess, grad_mask_consts=None):
    print('Printing the layer-wise weight counts to a file')

    # evaluate variables (get weights)
    vars = tf.trainable_variables()
    vals = sess.run(vars)

    if grad_mask_consts is not None:
        # check that the mask is working
        for idx, (var, val) in enumerate(zip(vars, vals)):
            # create np array of weights
            val_np = np.array(val)

            # mask out the un-pruned weights
            mask = np.array(sess.run(grad_mask_consts[var.name]))
            val_masked = val_np[mask == 0]

            # count how many of these are != 0
            leaked_pruned_weights = np.count_nonzero(val_masked)

            # count total number of pruned weights
            total = len(val_masked.flatten())

            # count total number of weights in this layer
            original = len(val_np.flatten())

            total_pruned = total - leaked_pruned_weights

            print('>>>\t{}: {} of the {} total weights have been pruned ({:.2f}% of original)'.
                  format(var.name, total_pruned, original, total_pruned / original * 100))
    else:
        # count total number of weights that are zero
        total_weights = 0
        total_zero = 0

        for idx, (var, val) in enumerate(zip(vars, vals)):
            # create np array of weights
            val_np = np.array(val)

            # count how many of these are != 0
            nonzero = np.count_nonzero(val_np)

            # count total number of pruned weights
            total = len(val_np.ravel())

            total_weights += total
            total_zero += total - nonzero

        print('>>>\t{} of the {} total weights have been pruned ({:.6f}% of original)'.
              format(total_zero, total_weights, total_zero / total_weights * 100))
        pass

########################################################################################################################
from keras.optimizers import Optimizer
import keras.backend as K
from keras.legacy import interfaces
# import tensorflow as tf

class CustomAdam(Optimizer):
    """Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".

    # References
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, grad_mask_consts=None, **kwargs):
        super(CustomAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.grad_mask_consts = grad_mask_consts

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        # if self.grad_mask_consts is not None:
        #     # Apply mask. orig_grads_and_vars is a list of tuples (gradient, variable).
        #     pruned_train_gradient = [
        #         (tf.multiply(tf.cast(self.grad_mask_consts[gv[1].name], tf.float32), gv[0]), gv[1]) for gv in grads]
        #
        #     opt_update = self.apply_gradients(
        #         grads, global_step=self.iterations)
        #     self.updates.append(opt_update)

        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            if self.grad_mask_consts is not None:
                print(p.name)
                print(self.grad_mask_consts[p.name])
                g = self.grad_mask_consts[p.name] * g
            # else:
            #     print('ok')
            #     exit(0)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(CustomAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# from tensorflow.python.eager import context
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import resource_variable_ops
# from tensorflow.python.ops import state_ops
# from tensorflow.python.training import optimizer
# from tensorflow.python.training import training_ops
# from tensorflow.python.util.tf_export import tf_export
#
# # from tensorflow.python.training.optimizer import
#
# GATE_OP = 1
#
#
# @tf_export("train.CustomAdamOptimizer")
# class CustomAdamOptimizer(optimizer.Optimizer):
#     """Optimizer that implements the Adam algorithm, with pruning enabled.
#     """
#
#     def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
#                  use_locking=False, name="Adam", grad_mask_consts=None):
#         """Construct a new Adam optimizer.
#
#         Initialization:
#
#         $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
#         $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
#         $$t := 0 \text{(Initialize timestep)}$$
#
#         The update rule for `variable` with gradient `g` uses an optimization
#         described at the end of section2 of the paper:
#
#         $$t := t + 1$$
#         $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
#
#         $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
#         $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
#         $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
#
#         The default value of 1e-8 for epsilon might not be a good default in
#         general. For example, when training an Inception network on ImageNet a
#         current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
#         formulation just before Section 2.1 of the Kingma and Ba paper rather than
#         the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
#         hat" in the paper.
#
#         The sparse implementation of this algorithm (used when the gradient is an
#         IndexedSlices object, typically because of `tf.gather` or an embedding
#         lookup in the forward pass) does apply momentum to variable slices even if
#         they were not used in the forward pass (meaning they have a gradient equal
#         to zero). Momentum decay (beta1) is also applied to the entire momentum
#         accumulator. This means that the sparse behavior is equivalent to the dense
#         behavior (in contrast to some momentum implementations which ignore momentum
#         unless a variable slice was actually used).
#
#         Args:
#           learning_rate: A Tensor or a floating point value.  The learning rate.
#           beta1: A float value or a constant float tensor.
#             The exponential decay rate for the 1st moment estimates.
#           beta2: A float value or a constant float tensor.
#             The exponential decay rate for the 2nd moment estimates.
#           epsilon: A small constant for numerical stability. This epsilon is
#             "epsilon hat" in the Kingma and Ba paper (in the formula just before
#             Section 2.1), not the epsilon in Algorithm 1 of the paper.
#           use_locking: If True use locks for update operations.
#           name: Optional name for the operations created when applying gradients.
#             Defaults to "Adam".
#
#         @compatibility(eager)
#         When eager execution is enabled, `learning_rate`, `beta1`, `beta2`, and
#         `epsilon` can each be a callable that takes no arguments and returns the
#         actual value to use. This can be useful for changing these values across
#         different invocations of optimizer functions.
#         @end_compatibility
#         """
#         super(CustomAdamOptimizer, self).__init__(use_locking, name)
#         self._lr = learning_rate
#         self._beta1 = beta1
#         self._beta2 = beta2
#         self._epsilon = epsilon
#
#         # Tensor versions of the constructor arguments, created in _prepare().
#         self._lr_t = None
#         self._beta1_t = None
#         self._beta2_t = None
#         self._epsilon_t = None
#
#         # Created in SparseApply if needed.
#         self._updated_lr = None
#
#         #######################################################
#         # Create the pruning dictionary
#         self.grad_mask_consts = grad_mask_consts
#
#     # def minimize(self, loss, global_step=None, var_list=None,
#     #              gate_gradients=GATE_OP, aggregation_method=None,
#     #              colocate_gradients_with_ops=False, name=None,
#     #              grad_loss=None):
#     #     """Add operations to minimize `loss` by updating `var_list`.
#     #
#     #     This method simply combines calls `compute_gradients()` and
#     #     `apply_gradients()`. If you want to process the gradient before applying
#     #     them call `compute_gradients()` and `apply_gradients()` explicitly instead
#     #     of using this function.
#     #
#     #     Args:
#     #       loss: A `Tensor` containing the value to minimize.
#     #       global_step: Optional `Variable` to increment by one after the
#     #         variables have been updated.
#     #       var_list: Optional list or tuple of `Variable` objects to update to
#     #         minimize `loss`.  Defaults to the list of variables collected in
#     #         the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
#     #       gate_gradients: How to gate the computation of gradients.  Can be
#     #         `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
#     #       aggregation_method: Specifies the method used to combine gradient terms.
#     #         Valid values are defined in the class `AggregationMethod`.
#     #       colocate_gradients_with_ops: If True, try colocating gradients with
#     #         the corresponding op.
#     #       name: Optional name for the returned operation.
#     #       grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
#     #
#     #     Returns:
#     #       An Operation that updates the variables in `var_list`.  If `global_step`
#     #       was not `None`, that operation also increments `global_step`.
#     #
#     #     Raises:
#     #       ValueError: If some of the variables are not `Variable` objects.
#     #
#     #     @compatibility(eager)
#     #     When eager execution is enabled, `loss` should be a Python function that
#     #     takes elements of `var_list` as arguments and computes the value to be
#     #     minimized. If `var_list` is None, `loss` should take no arguments.
#     #     Minimization (and gradient computation) is done with respect to the
#     #     elements of `var_list` if not None, else with respect to any trainable
#     #     variables created during the execution of the `loss` function.
#     #     `gate_gradients`, `aggregation_method`, `colocate_gradients_with_ops` and
#     #     `grad_loss` are ignored when eager execution is enabled.
#     #     @end_compatibility
#     #     """
#     #     grads_and_vars = self.compute_gradients(
#     #         loss, var_list=var_list, gate_gradients=gate_gradients,
#     #         aggregation_method=aggregation_method,
#     #         colocate_gradients_with_ops=colocate_gradients_with_ops,
#     #         grad_loss=grad_loss)
#     #
#     #     vars_with_grad = [v for g, v in grads_and_vars if g is not None]
#     #     if not vars_with_grad:
#     #         raise ValueError(
#     #             "No gradients provided for any variable, check your graph for ops"
#     #             " that do not support gradients, between variables %s and loss %s." %
#     #             ([str(v) for _, v in grads_and_vars], loss))
#     #
#     #     ################################################################################################################
#     #     # TODO: pruning step here
#     #     raise Exception('called!')
#     #     if self.grad_mask_consts is not None:
#     #
#     #         # Apply mask. orig_grads_and_vars is a list of tuples (gradient, variable).
#     #         pruned_train_gradient = [
#     #             (tf.multiply(tf.cast(self.grad_mask_consts[gv[1].name], tf.float32), gv[0]), gv[1]) for gv in grads_and_vars]
#     #
#     #         # Ask the optimizer to apply the masked gradients.
#     #         return self.apply_gradients(pruned_train_gradient, global_step=global_step)
#     #     else:
#     #         return self.apply_gradients(grads_and_vars, global_step=global_step,
#     #                                     name=name)
#
#     def _get_beta_accumulators(self):
#         with ops.init_scope():
#             if context.executing_eagerly():
#                 graph = None
#             else:
#                 graph = ops.get_default_graph()
#             return (self._get_non_slot_variable("beta1_power", graph=graph),
#                     self._get_non_slot_variable("beta2_power", graph=graph))
#
#     def _create_slots(self, var_list):
#         # Create the beta1 and beta2 accumulators on the same device as the first
#         # variable. Sort the var_list to make sure this device is consistent across
#         # workers (these need to go on the same PS, otherwise some updates are
#         # silently ignored).
#         first_var = min(var_list, key=lambda x: x.name)
#         self._create_non_slot_variable(initial_value=self._beta1,
#                                        name="beta1_power",
#                                        colocate_with=first_var)
#         self._create_non_slot_variable(initial_value=self._beta2,
#                                        name="beta2_power",
#                                        colocate_with=first_var)
#
#         # Create slots for the first and second moments.
#         for v in var_list:
#             self._zeros_slot(v, "m", self._name)
#             self._zeros_slot(v, "v", self._name)
#
#     def _prepare(self):
#         lr = self._call_if_callable(self._lr)
#         beta1 = self._call_if_callable(self._beta1)
#         beta2 = self._call_if_callable(self._beta2)
#         epsilon = self._call_if_callable(self._epsilon)
#
#         self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
#         self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
#         self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
#         self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
#
#     def _apply_dense(self, grad, var):
#         m = self.get_slot(var, "m")
#         v = self.get_slot(var, "v")
#         beta1_power, beta2_power = self._get_beta_accumulators()
#         return training_ops.apply_adam(
#             var, m, v,
#             math_ops.cast(beta1_power, var.dtype.base_dtype),
#             math_ops.cast(beta2_power, var.dtype.base_dtype),
#             math_ops.cast(self._lr_t, var.dtype.base_dtype),
#             math_ops.cast(self._beta1_t, var.dtype.base_dtype),
#             math_ops.cast(self._beta2_t, var.dtype.base_dtype),
#             math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
#             grad, use_locking=self._use_locking).op
#
#     def _resource_apply_dense(self, grad, var):
#         m = self.get_slot(var, "m")
#         v = self.get_slot(var, "v")
#         beta1_power, beta2_power = self._get_beta_accumulators()
#         return training_ops.resource_apply_adam(
#             var.handle, m.handle, v.handle,
#             math_ops.cast(beta1_power, grad.dtype.base_dtype),
#             math_ops.cast(beta2_power, grad.dtype.base_dtype),
#             math_ops.cast(self._lr_t, grad.dtype.base_dtype),
#             math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
#             math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
#             math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
#             grad, use_locking=self._use_locking)
#
#     def _apply_sparse_shared(self, grad, var, indices, scatter_add):
#         beta1_power, beta2_power = self._get_beta_accumulators()
#         beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
#         beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
#         lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
#         beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
#         beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
#         epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
#         lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
#         # m_t = beta1 * m + (1 - beta1) * g_t
#         m = self.get_slot(var, "m")
#         m_scaled_g_values = grad * (1 - beta1_t)
#         m_t = state_ops.assign(m, m * beta1_t,
#                                use_locking=self._use_locking)
#         with ops.control_dependencies([m_t]):
#             m_t = scatter_add(m, indices, m_scaled_g_values)
#         # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
#         v = self.get_slot(var, "v")
#         v_scaled_g_values = (grad * grad) * (1 - beta2_t)
#         v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
#         with ops.control_dependencies([v_t]):
#             v_t = scatter_add(v, indices, v_scaled_g_values)
#         v_sqrt = math_ops.sqrt(v_t)
#         var_update = state_ops.assign_sub(var,
#                                           lr * m_t / (v_sqrt + epsilon_t),
#                                           use_locking=self._use_locking)
#         return control_flow_ops.group(*[var_update, m_t, v_t])
#
#     def _apply_sparse(self, grad, var):
#         return self._apply_sparse_shared(
#             grad.values, var, grad.indices,
#             lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
#                 x, i, v, use_locking=self._use_locking))
#
#     def _resource_scatter_add(self, x, i, v):
#         with ops.control_dependencies(
#                 [resource_variable_ops.resource_scatter_add(
#                     x.handle, i, v)]):
#             return x.value()
#
#     def _resource_apply_sparse(self, grad, var, indices):
#         return self._apply_sparse_shared(
#             grad, var, indices, self._resource_scatter_add)
#
#     def _finish(self, update_ops, name_scope):
#         # Update the power accumulators.
#         with ops.control_dependencies(update_ops):
#             beta1_power, beta2_power = self._get_beta_accumulators()
#             with ops.colocate_with(beta1_power):
#                 update_beta1 = beta1_power.assign(
#                     beta1_power * self._beta1_t, use_locking=self._use_locking)
#                 update_beta2 = beta2_power.assign(
#                     beta2_power * self._beta2_t, use_locking=self._use_locking)
#         return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
#                                       name=name_scope)
