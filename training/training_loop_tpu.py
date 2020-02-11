# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Main training script."""

import os
import numpy as np
import tensorflow as tf
import tflex
import time
import dnnlib
import dnnlib.tflib as tflib
import traceback
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base
from training import train_runner
from training import imagenet_input
from training import networks_stylegan2

from absl import flags

import logging
import coloredlogs
logger = logging.getLogger(__name__)

def setup_logging(args):

  # Remove existing handlers at the root
  logging.getLogger().handlers = []

  coloredlogs.install(level=args.verbosity, logger=logger)

  for i in ['main_tpu', 'main_gpu', 'main_loop', 'utils', 'input', 'tensorflow', 'ops', 'BigGAN']:
    coloredlogs.install(level=args.verbosity, logger=logging.getLogger(i))

  logger.info(f"cmd args: {vars(args)}")



import sys

FLAGS = flags.FLAGS

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

flags.DEFINE_bool(
    'use_tpu', default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'master',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string('tpu_job_name', default=None, help='The tpu worker name.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific flags
flags.DEFINE_string(
    'data_dir', default=FAKE_DATA_DIR,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_integer(
    'resnet_depth', default=50,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))

flags.DEFINE_string(
    'mode', default='in_memory_eval',
    help='One of {"train_and_eval", "train", "eval"}.')

flags.DEFINE_integer(
    'train_steps', default=112590,
    help=('The number of steps to use for training. Default is 112590 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')

flags.DEFINE_integer(
    'steps_per_eval', default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_bool(
    'skip_host_call',
    default=True,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_parallel_calls', default=64,
    help=('Cycle length of the parallel interleave in tf.data.dataset.'))

flags.DEFINE_integer(
    'num_prefetch_threads',
    default=16,
    help=('Number of prefetch threads in CPU for the input pipeline'))

flags.DEFINE_bool(
    'prefetch_depth_auto_tune',
    default=True,
    help=('Number of prefetch threads in CPU for the input pipeline'))

flags.DEFINE_integer(
    'num_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string(
    'bigtable_project', None,
    'The Cloud Bigtable project.  If None, --gcp_project will be used.')
flags.DEFINE_string(
    'bigtable_instance', None,
    'The Cloud Bigtable instance to load data from.')
flags.DEFINE_string(
    'bigtable_table', 'imagenet',
    'The Cloud Bigtable table to load data from.')
flags.DEFINE_string(
    'bigtable_train_prefix', 'train_',
    'The prefix identifying training rows.')
flags.DEFINE_string(
    'bigtable_eval_prefix', 'validation_',
    'The prefix identifying evaluation rows.')
flags.DEFINE_string(
    'bigtable_column_family', 'tfexample',
    'The column family storing TFExamples.')
flags.DEFINE_string(
    'bigtable_column_qualifier', 'example',
    'The column name storing TFExamples.')

flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

# TODO(chrisying): remove this flag once --transpose_tpu_infeed flag is enabled
# by default for TPU
flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_string(
    'precision', default='float32',
    help=('Precision to use; one of: {bfloat16, float32}'))

flags.DEFINE_float(
    'base_learning_rate', default=0.1,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'weight_decay', default=1e-4,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.0,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                     'which the global step information is logged.')

flags.DEFINE_bool('enable_lars',
                  default=False,
                  help=('Enable LARS optimizer for large batch training.'))

flags.DEFINE_float('poly_rate', default=0.0,
                   help=('Set LARS/Poly learning rate.'))

flags.DEFINE_bool(
    'use_cache', default=True, help=('Enable cache for training input.'))
flags.DEFINE_bool(
    'cache_decoded_image', default=False, help=('Cache decoded images.'))

flags.DEFINE_bool(
    'use_async_checkpointing', default=False, help=('Enable async checkpoint'))
flags.DEFINE_float(
    'stop_threshold', default=0.759, help=('Stop threshold for MLPerf.'))

flags.DEFINE_bool(
    'use_eval_runner', default=True, help=('Bypass estimator on eval.'))

flags.DEFINE_bool(
    'use_train_runner', default=False, help=('Bypass estimator on train.'))

flags.DEFINE_integer(
    'tpu_cores_per_host', default=8, help=('Number of TPU cores per host.'))

flags.DEFINE_integer('image_size', 224, 'The input image size.')

flags.DEFINE_integer(
    'distributed_group_size',
    default=1,
    help=('When set to > 1, it will enable distributed batch normalization'))

# Learning rate schedule
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

flags.DEFINE_boolean(
    'output_summaries',
    default=False,
    help=('When set to true, outputs tensorboard logs'))

flags.DEFINE_boolean(
    'enable_auto_tracing',
    default=False,
    help=('When set to true traces collected from worker-0 on every run'))


args = sys.argv[1:]
if '--' in args:
    args = [sys.argv[0]] + args[args.index('--') + 1:]
else:
    args = [sys.argv[0]]
FLAGS_ = flags.FLAGS(args)

setup_logging(flags.FLAGS)

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, labels, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('DynamicRange'):
        x = tf.cast(x, tf.float32)
        x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            x = tf.where(tf.random_uniform([tf.shape(x)[0]]) < 0.5, x, tf.reverse(x, [3]))
    with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
        s = tf.shape(x)
        y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
        y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
        y = tf.tile(y, [1, 1, 1, 2, 1, 2])
        y = tf.reshape(y, [-1, s[1], s[2], s[3]])
        x = tflib.lerp(x, y, lod - tf.floor(lod))
    with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
        s = tf.shape(x)
        factor = tf.cast(2 ** tf.floor(lod), tf.int32)
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x, labels

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    lod_initial_resolution  = None,     # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_size_base     = 32,       # Global minibatch size.
    minibatch_size_dict     = {},       # Resolution-specific overrides.
    minibatch_gpu_base      = 4,        # Number of samples processed at a time by one GPU.
    minibatch_gpu_dict      = {},       # Resolution-specific overrides.
    G_lrate_base            = 0.002,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.002,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 4,        # Default interval of progress snapshots.
    tick_kimg_dict          = {8:28, 16:24, 32:20, 64:16, 128:12, 256:8, 512:6, 1024:4}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    if lod_initial_resolution is None:
        s.lod = 0.0
    else:
        s.lod = training_set.resolution_log2
        s.lod -= np.floor(np.log2(lod_initial_resolution))
        s.lod -= phase_idx
        if lod_transition_kimg > 0:
            s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch_size = minibatch_size_dict.get(s.resolution, minibatch_size_base)
    s.minibatch_gpu = minibatch_gpu_dict.get(s.resolution, minibatch_gpu_base)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s


def resnet_model_fn(features, labels, mode, params):
  """The model_fn for ResNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  #import pdb; pdb.set_trace()
  if isinstance(features, dict):
    features = features['feature']

  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU/TPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC.
  if FLAGS.data_format == 'channels_first':
    assert not FLAGS.transpose_input    # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])

  if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
    if FLAGS.train_batch_size // FLAGS.num_cores > 8:
      features = tf.reshape(features,
                            [FLAGS.image_size, FLAGS.image_size, 3, -1])
      features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC
    else:
      features = tf.reshape(features,
                            [FLAGS.image_size, FLAGS.image_size, -1, 3])
      features = tf.transpose(features, [2, 0, 1, 3])  # HWNC to NHWC

  # Normalize the image to zero mean and unit variance.
  #features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
  #features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

  # This nested function allows us to avoid duplicating the logic which
  # builds the network, for different values of --precision.
  training_set = params['training_set']
  G_args = params['G_args']
  D_args = params['D_args']
  G_loss_args = params['G_loss_args']
  D_loss_args = params['D_loss_args']

  def build_network():
    print('Constructing networks...')
    #import pdb; pdb.set_trace()
    minibatch_size = features.shape.as_list()[0]
    G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1],
                      label_size=training_set.label_size, **G_args)
    D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1],
                      label_size=training_set.label_size, **D_args)
    G.print_layers(); D.print_layers()
    #import pdb; pdb.set_trace()
    if False:
      latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
      labels2 = training_set.get_random_labels_tf(minibatch_size)
      fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels2, is_training=True, return_dlatents=True)
      fake_scores_out = D.get_output_for(fake_images_out, labels2, is_training=True)
      loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    #G = networks_stylegan2.G_main(features, labels)
    #Gs = G.clone('Gs')
    Gs = tflib.Network('Gs', num_channels=training_set.shape[0], resolution=training_set.shape[1],
                      label_size=training_set.label_size, **G_args)
    r = {'G': G, 'D': D, 'Gs': Gs, 'minibatch_size': minibatch_size}
    r = dnnlib.EasyDict(r)
    #import pdb; pdb.set_trace()
    return r

    #network = resnet_model.resnet_v1(
    #    resnet_depth=FLAGS.resnet_depth,
    #    num_classes=FLAGS.num_label_classes,
    #    data_format=FLAGS.data_format)
    #return network(
    #    inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  if FLAGS.precision == 'bfloat16':
    with tf.contrib.tpu.bfloat16_scope():
      logits = build_network()
    #logits = tf.cast(logits, tf.float32)
  elif FLAGS.precision == 'float32':
    logits = build_network()

  #import pdb; pdb.set_trace()

  """
  optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate,
    momentum=FLAGS.momentum,
    use_nesterov=True)

  if FLAGS.use_tpu:
    # When using TPU, wrap the optimizer with CrossShardOptimizer which
    # handles synchronization details between different TPU cores. To the
    # user, this should look like regular synchronous training.
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  # Batch normalization requires UPDATE_OPS to be added as a dependency to
  # the train operation.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step)
  """

  # Setup training inputs.
  print('Building TensorFlow graph...')
  G_smoothing_kimg = params['G_smoothing_kimg']
  with tf.name_scope('Inputs'):#, tflex.device('/cpu:0'):
    #lod_in = tf.placeholder(tf.float32, name='lod_in', shape=[])
    lod_in = 0.0
    #lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
    lrate_in = 0.002
    #minibatch_size_in = tf.placeholder(tf.int32, name='minibatch_size_in', shape=[])
    #minibatch_gpu_in = tf.placeholder(tf.int32, name='minibatch_gpu_in', shape=[])
    minibatch_gpu_in = logits.minibatch_size
    #num_gpus = FLAGS.num_cores // FLAGS.tpu_cores_per_host
    num_gpus = FLAGS.num_cores
    minibatch_size_in = FLAGS.num_cores
    minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
    minibatch_multiplier = 1.0
    Gs_beta = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32),
                            G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

  #import pdb; pdb.set_trace()

  # Setup optimizers.
  G_opt_args = params['G_opt_args']
  D_opt_args = params['D_opt_args']
  G_reg_interval = params['G_reg_interval']
  D_reg_interval = params['D_reg_interval']
  lazy_regularization = params['lazy_regularization']
  #lazy_regularization = False
  G_opt_args = dict(G_opt_args)
  D_opt_args = dict(D_opt_args)
  for args, reg_interval in [(G_opt_args, G_reg_interval), (D_opt_args, D_reg_interval)]:
    args['minibatch_multiplier'] = minibatch_multiplier
    args['learning_rate'] = lrate_in
    if lazy_regularization:
      mb_ratio = reg_interval / (reg_interval + 1)
      args['learning_rate'] *= mb_ratio
      if 'beta1' in args: args['beta1'] **= mb_ratio
      if 'beta2' in args: args['beta2'] **= mb_ratio
  def makeopt(**kws):
    if 'minibatch_multiplier' in kws:
      kws.pop('minibatch_multiplier')
    if 'share' in kws:
      kws.pop('share')
    #return tflib.Optimizer(**kws)
    opt = tf.train.AdamOptimizer(**kws)
    if FLAGS.use_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    return opt
  G_opt = makeopt(name='TrainG', **G_opt_args)
  D_opt = makeopt(name='TrainD', **D_opt_args)
  G_reg_opt = makeopt(name='RegG', share=G_opt, **G_opt_args)
  D_reg_opt = makeopt(name='RegD', share=D_opt, **D_opt_args)

  #import pdb; pdb.set_trace()

  Gs = logits.Gs
  G = G_gpu = logits.G
  D = D_gpu = logits.D

  minibatch_size = minibatch_gpu_in = logits.minibatch_size
  reals_read = tf.transpose(features, [0, 3, 1, 2])
  labels2 = training_set.get_random_labels_tf(minibatch_size)
  #labels_read = labels
  labels_read = labels2

  # Evaluate loss functions.
  lod_assign_ops = []
  if 'lod' in G_gpu.vars: lod_assign_ops += [tf.assign(G_gpu.vars['lod'], lod_in)]
  if 'lod' in D_gpu.vars: lod_assign_ops += [tf.assign(D_gpu.vars['lod'], lod_in)]
  with tf.control_dependencies(lod_assign_ops):
    with tf.name_scope('G_loss'):
      G_loss, G_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set,
                                                    minibatch_size=minibatch_gpu_in, **G_loss_args)
    with tf.name_scope('D_loss'):
      D_loss, D_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set,
                                                    minibatch_size=minibatch_gpu_in, reals=reals_read,
                                                    labels=labels_read, **D_loss_args)
  #import pdb; pdb.set_trace()

  def grad(opt, loss, train_vars):
    #opt_grads = tf.gradients(loss, train_vars)
    #opt_grads = list(zip(opt_grads, train_vars))
    #return opt_grads
    return opt.minimize(loss, var_list=train_vars, global_step=tf.train.get_global_step())

  # Register gradients.
  G_reg_op = None
  D_reg_op = None
  if not lazy_regularization:
    if G_reg is not None: G_loss += G_reg
    if D_reg is not None: D_loss += D_reg
  elif False:
    if G_reg is not None: G_reg_opt.register_gradients(tf.reduce_mean(G_reg * G_reg_interval), G_gpu.trainables)
    if D_reg is not None: D_reg_opt.register_gradients(tf.reduce_mean(D_reg * D_reg_interval), D_gpu.trainables)
  else:
    if G_reg is not None: G_reg_op = grad(G_reg_opt, tf.reduce_mean(G_reg * G_reg_interval), list(G_gpu.trainables.values()))
    if D_reg is not None: D_reg_op = grad(D_reg_opt, tf.reduce_mean(D_reg * D_reg_interval), list(D_gpu.trainables.values()))
  if False:
    G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
    D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)

    # Setup training ops.
    #data_fetch_op = tf.group(*data_fetch_ops)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    G_reg_op = G_reg_opt.apply_updates(allow_no_op=True)
    D_reg_op = D_reg_opt.apply_updates(allow_no_op=True)
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)
  else:
    G_loss_op = tf.reduce_mean(G_loss)
    D_loss_op = tf.reduce_mean(D_loss)
    G_train_op = grad(G_opt, G_loss_op, list(G_gpu.trainables.values()))
    D_train_op = grad(D_opt, D_loss_op, list(D_gpu.trainables.values()))
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)

  ops = []
  ops.append(G_train_op)
  if G_reg_op is not None:
    ops.append(G_reg_op)
  ops.append(D_train_op)
  if D_reg_op is not None:
    ops.append(D_reg_op)
  ops.append(Gs_update_op)
  train_op = tf.group(*ops)
  #import pdb; pdb.set_trace()

  predictions = {
    #"fake_image": predict_fake_images,
    "labels": labels,
  }
  metric_fn = None
  metric_fn_args = None

  loss = G_loss_op + D_loss_op
  return loss, train_op, predictions, metric_fn, metric_fn_args

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']   # pylint: disable=unused-variable

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=FLAGS.label_smoothing)

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_or_create_global_step()
    steps_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
    current_epoch = (tf.cast(global_step, tf.float32) /
                     steps_per_epoch)

    mlp_log.mlperf_print(
        'model_bn_span',
        FLAGS.distributed_group_size *
        (FLAGS.train_batch_size // FLAGS.num_cores))

    # Choose between LARS or momentum.
    if FLAGS.enable_lars:
      learning_rate = 0.0
      mlp_log.mlperf_print('opt_name', 'lars')
      optimizer = lars_util.init_lars_optimizer(current_epoch)
    else:
      mlp_log.mlperf_print('opt_name', 'sgd')
      learning_rate = learning_rate_schedule(current_epoch)
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=FLAGS.momentum,
          use_nesterov=True)
      mlp_log.mlperf_print('opt_momentum', FLAGS.momentum)
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if not FLAGS.skip_host_call:
      def host_call_fn(gs, loss, lr, ce):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with summary.create_file_writer(
            FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
          with summary.always_record_summaries():
            summary.scalar('loss', loss[0], step=gs)
            summary.scalar('learning_rate', lr[0], step=gs)
            summary.scalar('current_epoch', ce[0], step=gs)

            return summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits):
      """Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      # Mask out padded data with label -1.
      weights = tf.where(labels < 0, tf.zeros_like(labels),
                         tf.ones_like(labels))
      labels = tf.where(labels < 0, tf.zeros_like(labels), labels)
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions, weights=weights)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5, weights=weights)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics)


#----------------------------------------------------------------------------
# Main training script.

def training_loop_tpu(
    G_args                  = {},       # Options for generator network.
    D_args                  = {},       # Options for discriminator network.
    G_opt_args              = {},       # Options for generator optimizer.
    D_opt_args              = {},       # Options for discriminator optimizer.
    G_loss_args             = {},       # Options for generator loss.
    D_loss_args             = {},       # Options for discriminator loss.
    dataset_args            = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    data_dir                = None,     # Directory to load datasets from.
    G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    lazy_regularization     = True,     # Perform regularization as a separate training step?
    G_reg_interval          = 4,        # How often the perform regularization for G? Ignored if lazy_regularization=False.
    D_reg_interval          = 16,       # How often the perform regularization for D? Ignored if lazy_regularization=False.
    reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = only save 'networks-final.pkl'.
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_pkl              = None,     # Network pickle to resume training from, None = train from scratch.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0,      # Assumed wallclock time at the beginning. Affects reporting.
    resume_with_new_nets    = False):   # Construct new networks according to G_args and D_args before resuming training?

    if resume_pkl is None and 'RESUME_PKL' in os.environ:
        resume_pkl = os.environ['RESUME_PKL']
    if resume_kimg <= 0.0 and 'RESUME_KIMG' in os.environ:
        resume_kimg = float(os.environ['RESUME_KIMG'])
    if resume_time <= 0.0 and 'RESUME_TIME' in os.environ:
        resume_time = float(os.environ['RESUME_TIME'])

    # Initialize dnnlib and TensorFlow.
    #tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # Load training set.
    training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir), verbose=True, **dataset_args)
    #grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(training_set, **grid_args)
    #misc.save_image_grid(grid_reals, dnnlib.make_run_dir_path('reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)
    import pdb; pdb.set_trace()

    assert FLAGS.precision == 'bfloat16' or FLAGS.precision == 'float32', (
        'Invalid value for --precision flag; must be bfloat16 or float32.')
    tf.logging.info('Precision: %s', FLAGS.precision)
    use_bfloat16 = FLAGS.precision == 'bfloat16'

    # Input pipelines are slightly different (with regards to shuffling and
    # preprocessing) between training and evaluation.
    if FLAGS.data_dir == FAKE_DATA_DIR:
        tf.logging.info('Using fake dataset.')
    else:
        tf.logging.info('Using dataset: %s', FLAGS.data_dir)
    FLAGS.image_size = dataset_args.resolution
    imagenet_train, imagenet_eval = [
        imagenet_input.ImageNetInput(
            is_training=is_training,
            data_dir=FLAGS.data_dir,
            transpose_input=FLAGS.transpose_input,
            cache=FLAGS.use_cache and is_training,
            image_size=FLAGS.image_size,
            num_parallel_calls=FLAGS.num_parallel_calls,
            num_cores=FLAGS.num_prefetch_threads,
            prefetch_depth_auto_tune=FLAGS.prefetch_depth_auto_tune,
            use_bfloat16=use_bfloat16) for is_training in [True, False]
    ]

    #iterations_per_loop=157
    iterations_per_loop=1
    train_steps=2983
    trunner = train_runner.TrainRunner(iterations=iterations_per_loop, train_steps=train_steps)
    params = {'batch_size': FLAGS.train_batch_size,
              'G_args': G_args, 'D_args': D_args,
              'G_opt_args': G_opt_args, 'D_opt_args': D_opt_args,
              'G_loss_args': G_loss_args, 'D_loss_args': D_loss_args,
              'training_set': training_set,
              'G_smoothing_kimg': G_smoothing_kimg,
              'G_reg_interval': G_reg_interval,
              'D_reg_interval': D_reg_interval,
              'lazy_regularization': lazy_regularization,
              }
    params = dnnlib.EasyDict(params)
    def tpu_model_fn(features, labels, mode, params):
      loss, train_op, predictions, metric_fn, metric_fn_args = resnet_model_fn(features, labels, mode, params)

      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

      if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(
            metric_fn,
            metric_fn_args
          )
        )

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
      assert False

    trunner.initialize(imagenet_train.input_fn, tpu_model_fn, params)

    import pdb; pdb.set_trace()
    trunner.train()

    # Construct or load networks.
    with tflex.device('/gpu:0'):
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
            Gs = G.clone('Gs')
        if resume_pkl is not None:
            print('Loading networks from "%s"...' % resume_pkl)
            rG, rD, rGs = misc.load_pkl(resume_pkl)
            if resume_with_new_nets: G.copy_vars_from(rG); D.copy_vars_from(rD); Gs.copy_vars_from(rGs)
            else: G = rG; D = rD; Gs = rGs

    # Print layers and generate initial image snapshot.
    G.print_layers(); D.print_layers()
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set=training_set, **sched_args)
    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
    grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch_gpu)
    misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes_init.png'), drange=drange_net, grid_size=grid_size)

    def save_image_grid(latents, grid_size, filename):
        grid_fakes = Gs.run(latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch_gpu)
        misc.save_image_grid(grid_fakes, filename, drange=drange_net, grid_size=grid_size)

    tflex.save_image_grid = save_image_grid

    def save_image_grid_command(randomize=False):
        if randomize:
            tflex.latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
        if not hasattr(tflex, 'latents'):
            tflex.latents = grid_latents
        if randomize or not hasattr(tflex, 'grid_filename'):
            tflex.grid_filename = dnnlib.make_run_dir_path('grid%06d.png' % int(time.time()))
        use_grid_size = (2, 2)
        latents = tflex.latents[:np.prod(use_grid_size)]
        tflex.save_image_grid(latents, use_grid_size, tflex.grid_filename)
        print('Saved ' + tflex.grid_filename)

    tflex.save_image_grid_command = save_image_grid_command

    @tflex.register_command
    def image_grid():
        tflex.save_image_grid_command(randomize=True)

    @tflex.register_command
    def resave_image_grid():
        tflex.save_image_grid_command(randomize=False)

    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tflex.device('/cpu:0'):
        lod_in               = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in             = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_size_in    = tf.placeholder(tf.int32, name='minibatch_size_in', shape=[])
        minibatch_gpu_in     = tf.placeholder(tf.int32, name='minibatch_gpu_in', shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
        Gs_beta              = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    # Setup optimizers.
    G_opt_args = dict(G_opt_args)
    D_opt_args = dict(D_opt_args)
    for args, reg_interval in [(G_opt_args, G_reg_interval), (D_opt_args, D_reg_interval)]:
        args['minibatch_multiplier'] = minibatch_multiplier
        args['learning_rate'] = lrate_in
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            args['learning_rate'] *= mb_ratio
            if 'beta1' in args: args['beta1'] **= mb_ratio
            if 'beta2' in args: args['beta2'] **= mb_ratio
    G_opt = tflib.Optimizer(name='TrainG', **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', **D_opt_args)
    G_reg_opt = tflib.Optimizer(name='RegG', share=G_opt, **G_opt_args)
    D_reg_opt = tflib.Optimizer(name='RegD', share=D_opt, **D_opt_args)

    # Build training graph for each GPU.
    data_fetch_ops = []
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tflex.device('/gpu:%d' % gpu):

            # Create GPU-specific shadow copies of G and D.
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')

            # Fetch training data via temporary variables.
            with tf.name_scope('DataFetch'):
                sched = training_schedule(cur_nimg=int(resume_kimg*1000), training_set=training_set, **sched_args)
                reals_var = tf.Variable(name='reals', trainable=False, initial_value=tf.zeros([sched.minibatch_gpu] + training_set.shape))
                labels_var = tf.Variable(name='labels', trainable=False, initial_value=tf.zeros([sched.minibatch_gpu, training_set.label_size]))
                reals_write, labels_write = training_set.get_minibatch_tf()
                reals_write, labels_write = process_reals(reals_write, labels_write, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
                reals_write = tf.concat([reals_write, reals_var[minibatch_gpu_in:]], axis=0)
                labels_write = tf.concat([labels_write, labels_var[minibatch_gpu_in:]], axis=0)
                data_fetch_ops += [tf.assign(reals_var, reals_write)]
                data_fetch_ops += [tf.assign(labels_var, labels_write)]
                reals_read = reals_var[:minibatch_gpu_in]
                labels_read = labels_var[:minibatch_gpu_in]

            # Evaluate loss functions.
            lod_assign_ops = []
            if 'lod' in G_gpu.vars: lod_assign_ops += [tf.assign(G_gpu.vars['lod'], lod_in)]
            if 'lod' in D_gpu.vars: lod_assign_ops += [tf.assign(D_gpu.vars['lod'], lod_in)]
            with tf.control_dependencies(lod_assign_ops):
                with tf.name_scope('G_loss'):
                    G_loss, G_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set, minibatch_size=minibatch_gpu_in, **G_loss_args)
                with tf.name_scope('D_loss'):
                    D_loss, D_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_gpu_in, reals=reals_read, labels=labels_read, **D_loss_args)

            # Register gradients.
            if not lazy_regularization:
                if G_reg is not None: G_loss += G_reg
                if D_reg is not None: D_loss += D_reg
            else:
                if G_reg is not None: G_reg_opt.register_gradients(tf.reduce_mean(G_reg * G_reg_interval), G_gpu.trainables)
                if D_reg is not None: D_reg_opt.register_gradients(tf.reduce_mean(D_reg * D_reg_interval), D_gpu.trainables)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)

    # Setup training ops.
    data_fetch_op = tf.group(*data_fetch_ops)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    G_reg_op = G_reg_opt.apply_updates(allow_no_op=True)
    D_reg_op = D_reg_opt.apply_updates(allow_no_op=True)
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)

    # Finalize graph.
    if tflex.has_gpu():
        with tflex.device('/gpu:0'):
            try:
                peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
            except tf.errors.NotFoundError:
                peak_gpu_mem_op = tf.constant(0)
    else:
        peak_gpu_mem_op = None
    tflib.init_uninitialized_vars()

    print('Initializing logs...')
    summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0
    while cur_nimg < total_kimg * 1000:
        if tflex.state.noisy: print('cur_nimg', cur_nimg, total_kimg)
        if dnnlib.RunContext.get().should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        training_set.configure(sched.minibatch_gpu, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        feed_dict = {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_size_in: sched.minibatch_size, minibatch_gpu_in: sched.minibatch_gpu}
        for _repeat in range(minibatch_repeats):
            if tflex.state.noisy: print('_repeat', _repeat)
            rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
            run_G_reg = (lazy_regularization and running_mb_counter % G_reg_interval == 0)
            run_D_reg = (lazy_regularization and running_mb_counter % D_reg_interval == 0)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            if len(rounds) == 1:
                if tflex.state.noisy: print('G_train_op', 'fast path')
                tflib.run([G_train_op, data_fetch_op], feed_dict)
                if run_G_reg:
                    tflib.run(G_reg_op, feed_dict)
                if tflex.state.noisy: print('D_train_op', 'fast path')
                tflib.run([D_train_op, Gs_update_op], feed_dict)
                if run_D_reg:
                    tflib.run(D_reg_op, feed_dict)

            # Slow path with gradient accumulation.
            else:
                for _round in rounds:
                    if tflex.state.noisy: print('G_train_op', 'slow path')
                    tflib.run(G_train_op, feed_dict)
                if run_G_reg:
                    for _round in rounds:
                        if tflex.state.noisy: print('G_reg_op', 'slow path')
                        tflib.run(G_reg_op, feed_dict)
                if tflex.state.noisy: print('G_update_op', 'slow path')
                tflib.run(Gs_update_op, feed_dict)
                for _round in rounds:
                    if tflex.state.noisy: print('data_fetch_op', 'slow path')
                    tflib.run(data_fetch_op, feed_dict)
                    if tflex.state.noisy: print('D_train_op', 'slow path')
                    tflib.run(D_train_op, feed_dict)
                if run_D_reg:
                    for _round in rounds:
                        if tflex.state.noisy: print('D_reg_op', 'slow path')
                        tflib.run(D_reg_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()

            def report_progress_command():
                total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time
                tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
                tick_time = dnnlib.RunContext.get().get_time_since_last_update()
                print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f' % (
                    autosummary('Progress/tick', cur_tick),
                    autosummary('Progress/kimg', cur_nimg / 1000.0),
                    autosummary('Progress/lod', sched.lod),
                    autosummary('Progress/minibatch', sched.minibatch_size),
                    dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                    autosummary('Timing/sec_per_tick', tick_time),
                    autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                    autosummary('Timing/maintenance_sec', maintenance_time),
                    autosummary('Resources/peak_gpu_mem_gb', (peak_gpu_mem_op.eval() if peak_gpu_mem_op is not None else 0) / 2**30)))
                autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
                autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            if not hasattr(tflex, 'report_progress_command'):
                tflex.report_progress_command = report_progress_command

            @tflex.register_command
            def report_progress():
                tflex.report_progress_command()

            def save_command():
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl)
                metrics.run(pkl, run_dir=dnnlib.make_run_dir_path(), data_dir=dnnlib.convert_path(data_dir), num_gpus=num_gpus, tf_config=tf_config)

            if not hasattr(tflex, 'save_command'):
                tflex.save_command = save_command

            @tflex.register_command
            def save():
                tflex.save_command()

            try:
              # Report progress.
              tflex.report_progress_command()
              tick_start_nimg = cur_nimg

              # Save snapshots.
              if image_snapshot_ticks is not None and (cur_tick % image_snapshot_ticks == 0 or done):
                  grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch_gpu)
                  misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
              if network_snapshot_ticks is not None and cur_tick > 0 and (cur_tick % network_snapshot_ticks == 0 or done):
                  tflex.save_command()

              # Update summaries and RunContext.
              metrics.update_autosummaries()
              tflib.autosummary.save_summaries(summary_log, cur_nimg)
              dnnlib.RunContext.get().update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
              maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time
            except:
              traceback.print_exc()

    # Save final snapshot.
    misc.save_pkl((G, D, Gs), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()

#----------------------------------------------------------------------------
