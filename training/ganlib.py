import tensorflow as tf
import pickle
import dnnlib.tflib as tflib
import dnnlib
import tflex

def load_model(
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', # karras2019stylegan-ffhq-1024x1024.pkl
    session = None,
    cache_dir = 'cache'):
  session = session or tflex.get_default_session()
  with session.as_default():
    from training.networks_stylegan import G_style, D_basic
    with dnnlib.util.open_url(url, cache_dir=cache_dir) as f:
      _G, _D, _Gs = pickle.load(f)
    G = tflib.Network(_G.name, G_style, **_G.static_kwargs)
    G.copy_vars_from(_G)
    D = tflib.Network(_D.name, D_basic, **_D.static_kwargs)
    D.copy_vars_from(_D)
    Gs = tflib.Network(_Gs.name, G_style, **_Gs.static_kwargs)
    Gs.copy_vars_from(_Gs)
    return G, D, Gs

def load_perceptual(
    url = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', # vgg16_zhang_perceptual.pkl
    session = None,
    cache_dir = '.stylegan2_cache'):
  session = session or tflex.get_default_session()

  #with tf.Session().as_default() as sess:
  with dnnlib.util.open_url(url, cache_dir=cache_dir) as f:
    _P = pickle.load(f)
  import pdb; pdb.set_trace()

  from training.vgg16_zhang_perceptual import lpips_network
  P = dnnlib.EasyDict()
  P.name = 'vgg16_perceptual_distance'
  P.static_kwargs = {}
  P = tflib.Network(P.name, lpips_network, **P.static_kwargs)
  with session.as_default():
    P.copy_vars_from(_P)
    return P

