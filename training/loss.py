# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Loss functions."""

import os
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary, autofid

#----------------------------------------------------------------------------
# Logistic loss from the paper
# "Generative Adversarial Nets", Goodfellow et al. 2014

def G_logistic(G, D, opt, training_set, minibatch_size):
    _ = opt
    if training_set.precalc:
        fake_scores_out = training_set.fake_scores
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        labels = training_set.get_random_labels_tf(minibatch_size)
        fake_images_out = G.get_output_for(latents, labels, is_training=True)
        fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = -tf.nn.softplus(fake_scores_out) # log(1-sigmoid(fake_scores_out)) # pylint: disable=invalid-unary-operand-type
    autosummary('G_logistic_00/total_loss', loss)
    return loss, None

def G_logistic_ns(G, D, opt, training_set, minibatch_size):
    _ = opt
    if training_set.precalc:
        fake_scores_out = training_set.fake_scores
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        labels = training_set.get_random_labels_tf(minibatch_size)
        fake_images_out = G.get_output_for(latents, labels, is_training=True)
        fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    autosummary('G_logistic_ns_00/total_loss', loss)
    return loss, None

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels):
    _ = opt, training_set
    if training_set.precalc:
        fake_images_out, fake_scores_out = training_set.fake_images, training_set.fake_scores
        real_images_out, real_scores_out = training_set.real_images, training_set.real_scores
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        real_images_out = reals
        fake_images_out = G.get_output_for(latents, labels, is_training=True)
        real_scores_out = D.get_output_for(real_images_out, labels, is_training=True)
        fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('D_logistic_00/real_scores', real_scores_out)
    fake_scores_out = autosummary('D_logistic_01/fake_scores', fake_scores_out)
    loss = autosummary('D_logistic_00/fake_loss', tf.nn.softplus(fake_scores_out)) # -log(1-sigmoid(fake_scores_out))
    loss += autosummary('D_logistic_01/real_loss', tf.nn.softplus(-real_scores_out)) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    autosummary('D_logistic_02/total_loss', loss)
    autofid('D_logistic/images', real_images_out, fake_images_out)
    return loss, None

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    if training_set.precalc:
        fake_images_out, fake_scores_out = training_set.fake_images, training_set.fake_scores
        real_images_out, real_scores_out = training_set.real_images, training_set.real_scores
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        real_images_out = reals
        fake_images_out = G.get_output_for(latents, labels, is_training=True)
        real_scores_out = D.get_output_for(real_images_out, labels, is_training=True)
        fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = autosummary('D_logistic_r1_00/fake_scores', fake_scores_out)
    real_scores_out = autosummary('D_logistic_r1_01/real_scores', real_scores_out)
    loss = autosummary('D_logistic_r1_00/fake_loss', tf.nn.softplus(fake_scores_out)) # -log(1-sigmoid(fake_scores_out))
    loss += autosummary('D_logistic_r1_01/real_loss', tf.nn.softplus(-real_scores_out)) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [real_images_out])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('D_logistic_r1_02/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
        autosummary('D_logistic_r1_02/reg_loss', reg)
    autosummary('D_logistic_r1_03/total_loss', loss + reg)
    autofid('D_logistic_r1/images', real_images_out, fake_images_out)
    return loss, reg

def D_logistic_r2(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    if training_set.precalc:
        fake_images_out, fake_scores_out = training_set.fake_images, training_set.fake_scores
        real_images_out, real_scores_out = training_set.real_images, training_set.real_scores
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        real_images_out = reals
        fake_images_out = G.get_output_for(latents, labels, is_training=True)
        real_scores_out = D.get_output_for(real_images_out, labels, is_training=True)
        fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
        fake_scores_out = autosummary('D_logistic_r2_00/fake_scores', fake_scores_out)
        real_scores_out = autosummary('D_logistic_r2_01/real_scores', real_scores_out)
    loss = autosummary('D_logistic_r2_00/fake_loss', tf.nn.softplus(fake_scores_out)) # -log(1-sigmoid(fake_scores_out))
    loss += autosummary('D_logistic_r2_01/real_loss', tf.nn.softplus(-real_scores_out)) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    with tf.name_scope('GradientPenalty'):
        fake_grads = tf.gradients(tf.reduce_sum(fake_scores_out), [fake_images_out])[0]
        gradient_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])
        gradient_penalty = autosummary('D_logistic_r2_02/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
        autosummary('D_logistic_r2_02/reg_loss', reg)
    autosummary('D_logistic_r2_03/total_loss', loss + reg)
    autofid('D_logistic_r2/images', real_images_out, fake_images_out)
    return loss, reg

#----------------------------------------------------------------------------
# WGAN loss from the paper
# "Wasserstein Generative Adversarial Networks", Arjovsky et al. 2017

def G_wgan(G, D, opt, training_set, minibatch_size):
    _ = opt
    if training_set.precalc:
        fake_scores_out = training_set.fake_scores
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        labels = training_set.get_random_labels_tf(minibatch_size)
        fake_images_out = G.get_output_for(latents, labels, is_training=True)
        fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = -fake_scores_out
    autosummary('G_wgan_00/total_loss', loss)
    return loss, None

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, wgan_epsilon=0.001):
    _ = opt, training_set
    if training_set.precalc:
        fake_images_out, fake_scores_out = training_set.fake_images, training_set.fake_scores
        real_images_out, real_scores_out = training_set.real_images, training_set.real_scores
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        real_images_out = reals
        fake_images_out = G.get_output_for(latents, labels, is_training=True)
        real_scores_out = D.get_output_for(real_images_out, labels, is_training=True)
        fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = autosummary('D_wgan_00/fake_score', fake_scores_out)
    real_scores_out = autosummary('D_wgan_01/real_score', real_scores_out)
    loss = autosummary('D_wgan_00/fake_loss', fake_scores_out)
    loss += autosummary('D_wgan_01/real_loss', -real_scores_out)
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('D_wgan_02/epsilon_penalty', tf.square(real_scores_out))
        loss += autosummary('D_wgan_02/penalty_loss', epsilon_penalty * wgan_epsilon)
    autosummary('D_wgan_03/total_loss', loss)
    autofid('D_wgan/images', real_images_out, fake_images_out)
    return loss, None

#----------------------------------------------------------------------------
# WGAN-GP loss from the paper
# "Improved Training of Wasserstein GANs", Gulrajani et al. 2017

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, wgan_lambda=10.0, wgan_epsilon=0.001, wgan_target=1.0):
    _ = opt, training_set
    if training_set.precalc:
        fake_images_out, fake_scores_out = training_set.fake_images, training_set.fake_scores
        real_images_out, real_scores_out = training_set.real_images, training_set.real_scores
        real_labels_out = training_set.real_labels
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        real_labels_out = labels
        real_images_out = reals
        fake_images_out = G.get_output_for(latents, real_labels_out, is_training=True)
        real_scores_out = D.get_output_for(real_images_out, real_labels_out, is_training=True)
        fake_scores_out = D.get_output_for(fake_images_out, real_labels_out, is_training=True)
    fake_scores_out = autosummary('D_wgan_gp_00/fake_scores', fake_scores_out)
    real_scores_out = autosummary('D_wgan_gp_01/real_scores', real_scores_out)
    loss = autosummary('D_wgan_gp_00/fake_loss', fake_scores_out)
    loss += autosummary('D_wgan_gp_01/real_loss', -real_scores_out)
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('D_wgan_gp_02/epsilon_penalty', tf.square(real_scores_out))
        loss += autosummary('D_wgan_gp_02/penalty_loss', epsilon_penalty * wgan_epsilon)

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(real_images_out, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = D.get_output_for(mixed_images_out, real_labels_out, is_training=True)
        mixed_scores_out = autosummary('D_wgan_gp_03/mixed_scores', mixed_scores_out)
        mixed_grads = tf.gradients(tf.reduce_sum(mixed_scores_out), [mixed_images_out])[0]
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('D_wgan_gp_03/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
        reg = gradient_penalty * (wgan_lambda / (wgan_target**2))
        autosummary('D_wgan_gp_03/gradient_penalty', gradient_penalty)
        autosummary('D_wgan_gp_03/reg_loss', reg)
    autosummary('D_wgan_gp_04/total_loss', loss + reg)
    autofid('D_wgan_gp/images', real_images_out, fake_images_out)
    return loss, reg

#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

def G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=None, pl_decay=0.01, pl_weight=2.0):
    if pl_minibatch_shrink is None:
        pl_minibatch_shrink = int(os.environ.get('PL_MINIBATCH_SHRINK', '1'))
    _ = opt
    if training_set.precalc:
        fake_images_out, fake_scores_out = training_set.fake_images, training_set.fake_scores
        fake_dlatents_out = training_set.fake_dlatents
    else:
        latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        labels = training_set.get_random_labels_tf(minibatch_size)
        fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
        fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    autosummary('G_logistic_ns_pathreg_00/fake_scores', fake_scores_out)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    autosummary('G_logistic_ns_pathreg_00/fake_loss', loss)

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = tf.maximum(1, minibatch_size // pl_minibatch_shrink)
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('G_logistic_ns_pathreg_01/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        # TODO: Should this use resource vars? Is this safe in a multi-core TPU setting?
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
            autosummary('G_logistic_ns_pathreg_01/pl_mean', pl_mean_var)
        pl_mean_delta = pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        if 'TPU_NAME' in os.environ and bool(int(os.environ.get('PL_CROSS_REPLICA', '0'))):
            pl_mean_delta = cross_replica_mean(pl_mean_delta)
        pl_mean = pl_mean_var + autosummary('G_logistic_ns_pathreg_01/pl_mean_delta', pl_mean_delta)
        pl_update = tf.assign(pl_mean_var, pl_mean)
        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('G_logistic_ns_pathreg_01/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight
        autosummary('G_logistic_ns_pathreg_01/reg_loss', reg)
    autosummary('G_logistic_ns_pathreg_02/total_loss', loss + reg)

    return loss, reg

#----------------------------------------------------------------------------


def cross_replica_concat(value, replica_id, num_replicas):
  """Reduce a concatenation of the `value` across TPU replicas.

  Args:
    value: Tensor to concatenate.
    replica_id: Integer tensor that indicates the index of the replica.
    num_replicas: Python integer, total number of replicas.

  Returns:
    Tensor of the same rank as value with first dimension `num_replicas`
    times larger.

  Raises:
    ValueError: If `value` is a scalar.
  """
  if value.shape.ndims < 1:
    raise ValueError("Value must have at least rank 1 but got {}.".format(
        value.shape.ndims))
  if num_replicas <= 1:
    return value
  with tf.name_scope(None, "tpu_cross_replica_concat"):
    # Mask is one hot encoded position of the core_index.
    mask = tf.to_float(tf.equal(tf.range(num_replicas), replica_id))
    # Expand dims with 1's to match rank of value.
    mask = tf.reshape(mask, [num_replicas] + [1] * value.shape.ndims)
    if value.dtype in {tf.bfloat16, tf.float32}:
      result = mask * value
    else:
      result = mask * tf.to_float(value)
    # Thanks to broadcasting now result is set only in the position pointed by
    # replica_id, the rest of the vector is set to 0's.
    # All these steps are basically implementing tf.scatter_nd which is missing
    # in TPU's backend since it doesn't support sparse operations.

    # Merge first 2 dimensions.
    # This is equivalent to (value.shape[0].value * num_replicas).
    # Using [-1] trick to support also scalar input.
    result = tf.reshape(result, [-1] + result.shape.as_list()[2:])
    # Each core set the "results" in position pointed by replica_id. When we now
    # sum across replicas we exchange the information and fill in local 0's with
    # values from other cores.
    result = tf.contrib.tpu.cross_replica_sum(result)
    # Now all the cores see exactly the same data.
    return tf.cast(result, dtype=value.dtype)


from tensorflow.contrib.tpu.python.tpu import tpu_function


def cross_replica_mean(inputs, group_size=None):
  """Calculates the average value of inputs tensor across TPU replicas."""
  num_replicas = tpu_function.get_tpu_context().number_of_shards
  if not group_size:
    group_size = num_replicas
  if group_size == 1:
    return inputs
  if group_size != num_replicas:
    group_assignment = []
    assert num_replicas % group_size == 0
    for g in range(num_replicas // group_size):
      replica_ids = [g * group_size + i for i in range(group_size)]
      group_assignment.append(replica_ids)
  else:
    group_assignment = None
  return tf.contrib.tpu.cross_replica_sum(inputs, group_assignment) / tf.cast(
      group_size, inputs.dtype)


