import tensorflow as tf

import functools


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn


@op_scope
def ti32(x):
  return tf.cast( x, tf.int32 )


@op_scope
def tf32(x):
  return tf.cast( x, tf.float32 )


@op_scope
def clamp(v, min=0., max=1.):
  return tf.clip_by_value(v, min, max)


@op_scope
def wrap(v, wrap_mode="reflect"):
  assert wrap_mode in ["clamp", "wrap", "reflect"]
  if wrap_mode == "wrap":
    return tf.math.floormod(v, 1.0)
  elif wrap_mode == "reflect":
    return 1.0 - tf.abs(tf.math.floormod(v, 2.0) - 1.0)
  elif wrap_mode == "clamp":
    return clamp(v)


@op_scope
def iround(u):
  return ti32(tf.math.floordiv(tf32(u), 1.0))


@op_scope
def tf_image_translate(images, x_offset, y_offset, data_format="NHWC", wrap_mode="reflect", static_batch=None):
  assert data_format in ["NHWC", "NCHW"]
  if data_format == "NCHW":
    images = tf.transpose(images, [0, 2, 3, 1]) # NCHW to NHWC
  if static_batch is not None:
    out = [tf_image_translate_1(images[i], x_offset[i], y_offset[i]) for i in range(static_batch)]
    out = tf.stack(out)
  else:
    def thunk(args):
      return [tf_image_translate_1(*args, data_format="NHWC", wrap_mode=wrap_mode)] + args[1:]
    out, _, _ = tf.map_fn(thunk, [images, x_offset, y_offset])
  if data_format == "NCHW":
    out = tf.transpose(out, [0, 3, 1, 2]) # NHWC to NCHW
  return out


@op_scope
def tf_image_translate_1(img, x_offset, y_offset, data_format="NHWC", wrap_mode="reflect"):
  # "NCHW" not implemented for now; handled by transpose in tf_image_translate
  #assert data_format in ["NHWC", "NCHW"]
  assert data_format in ["NHWC"]
  shape = tf.shape(img)
  if data_format == "NHWC":
    h, w, c = shape[0], shape[1], shape[2]
  else:
    c, h, w = shape[0], shape[1], shape[2]

  DUDX = 1.0 / tf32(w)
  DUDY = 0.0
  DVDX = 0.0
  DVDY = 1.0 / tf32(h)

  X1 = 0
  Y1 = 0
  U1 = x_offset
  V1 = y_offset

  # Calculate UV at screen origin.
  U = U1 - DUDX*tf32(X1) - DUDY*tf32(Y1)
  V = V1 - DVDX*tf32(X1) - DVDY*tf32(Y1)

  u  = tf.cumsum(tf.fill([h, w], DUDX), 1)
  u += tf.cumsum(tf.fill([h, w], DUDY), 0)
  u += U
  v  = tf.cumsum(tf.fill([h, w], DVDX), 1)
  v += tf.cumsum(tf.fill([h, w], DVDY), 0)
  v += V
  uv = tf.stack([v, u], 2)

  th, tw = h, w
  uv = clamp(iround(wrap(uv, "reflect") * [tw, th]), 0, [tw-1, th-1])
  color = tf.gather_nd(img, uv)
  return color


@op_scope
def tf_random_translate(imgs, x_strength=0.5, y_strength=0.5, data_format="NHWC", wrap_mode="reflect", static_batch=None):
  shape = [tf.shape(imgs)[0]]
  x_offset = tf.random.uniform(shape, minval=-x_strength, maxval=x_strength)
  y_offset = tf.random.uniform(shape, minval=-y_strength, maxval=y_strength)
  return tf_image_translate(imgs, x_offset, y_offset, data_format=data_format, wrap_mode=wrap_mode, static_batch=static_batch)


@op_scope
def tf_image_augment(imgs, data_format="NHWC", static_batch=None):
  imgs = tf_random_translate(imgs, data_format=data_format, static_batch=static_batch)
  return imgs

