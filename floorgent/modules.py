# vim: sts=2 sw=2 ts=2

import warnings

import numpy as np
import sonnet as snt
# Should really `import tensorflow.compat.v1 as tf`, but Visual Studio Code
# can't deal with the obvious version for some reason.
import tensorflow._api.v1.compat.v1 as tf
from tensorflow.python.framework import function
from tensorflow_probability import distributions as tfd
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

from .polylines import code_tokens
from .polylines import sequence_end_token


def add_activation(x, name='activation'):
    # For debugging, slows down graph execution a lot.
    #x = tf.identity(x, name=f'act_{name}')
    #tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    return



# Mostly copied from deepmind_research/polygen/modules.py.
# ------------------------- 8< --------------------------


def top_k_logits(logits, k):
  """Masks logits such that logits not in top-k are small."""
  if k == 0:
    return logits
  else:
    values, _ = tf.math.top_k(logits, k=k)
    k_largest = tf.reduce_min(values)
    logits = tf.where(tf.less_equal(logits, k_largest),
                      tf.ones_like(logits)*-1e9, logits)
    return logits


def top_p_logits(logits, p):
  """Masks logits using nucleus (top-p) sampling."""
  if p == 1:
    return logits
  logit_shape = tf.shape(logits)
  seq, dim = logit_shape[1], logit_shape[2]
  logits = tf.reshape(logits, [-1, dim])
  sort_indices = tf.argsort(logits, axis=-1, direction='DESCENDING')
  probs = tf.gather(tf.nn.softmax(logits), sort_indices, batch_dims=1)
  cumprobs = tf.cumsum(probs, axis=-1, exclusive=True)
  # The top 1 candidate always will not be masked.
  # This way ensures at least 1 indices will be selected.
  sort_mask = tf.cast(tf.greater(cumprobs, p), logits.dtype)
  batch_indices = tf.tile(
      tf.expand_dims(tf.range(tf.shape(logits)[0]), axis=-1), [1, dim])
  top_p_mask = tf.scatter_nd(
      tf.stack([batch_indices, sort_indices], axis=-1),
      sort_mask,
      tf.shape(logits))
  logits -= top_p_mask * 1e9
  return tf.reshape(logits, [-1, seq, dim])

_function_cache = {}  # For multihead_self_attention_memory_efficient


def multihead_self_attention_memory_efficient(x,
                                              bias,
                                              num_heads,
                                              head_size=None,
                                              cache=None,
                                              epsilon=1e-6,
                                              forget=True,
                                              test_vars=None,
                                              name=None):
  """Memory-efficient Multihead scaled-dot-product self-attention.

  Based on Tensor2Tensor version but adds optional caching.

  Returns multihead-self-attention(layer_norm(x))

  Computes one attention head at a time to avoid exhausting memory.

  If forget=True, then forget all forwards activations and recompute on
  the backwards pass.

  Args:
    x: a Tensor with shape [batch, length, input_size]
    bias: an attention bias tensor broadcastable to [batch, 1, length, length]
    num_heads: an integer
    head_size: an optional integer - defaults to input_size/num_heads
    cache: Optional dict containing tensors which are the results of previous
        attentions, used for fast decoding. Expects the dict to contain two
        keys ('k' and 'v'), for the initial call the values for these keys
        should be empty Tensors of the appropriate shape.
        'k' [batch_size, 0, key_channels] 'v' [batch_size, 0, value_channels]
    epsilon: a float, for layer norm
    forget: a boolean - forget forwards activations and recompute on backprop
    test_vars: optional tuple of variables for testing purposes
    name: an optional string

  Returns:
    A Tensor.
  """
  io_size = x.get_shape().as_list()[-1]
  if head_size is None:
    assert io_size % num_heads == 0
    head_size = io_size / num_heads

  def forward_internal(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
    """Forward function."""
    n = common_layers.layer_norm_compute(x, epsilon, norm_scale, norm_bias)
    wqkv_split = tf.unstack(wqkv, num=num_heads)
    wo_split = tf.unstack(wo, num=num_heads)
    y = 0
    if cache is not None:
      cache_k = []
      cache_v = []
    for h in range(num_heads):
      with tf.control_dependencies([y] if h > 0 else []):
        combined = tf.nn.conv1d(n, wqkv_split[h], 1, 'SAME')
        q, k, v = tf.split(combined, 3, axis=2)
        if cache is not None:
          k = tf.concat([cache['k'][:, h], k], axis=1)
          v = tf.concat([cache['v'][:, h], v], axis=1)
          cache_k.append(k)
          cache_v.append(v)
        o = common_attention.scaled_dot_product_attention_simple(
            q, k, v, attention_bias)
        y += tf.nn.conv1d(o, wo_split[h], 1, 'SAME')
    if cache is not None:
      cache['k'] = tf.stack(cache_k, axis=1)
      cache['v'] = tf.stack(cache_v, axis=1)
    return y

  key = (
      'multihead_self_attention_memory_efficient %s %s' % (num_heads, epsilon))
  if not forget:
    forward_fn = forward_internal
  elif key in _function_cache:
    forward_fn = _function_cache[key]
  else:

    @function.Defun(compiled=True)
    def grad_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias, dy):
      """Custom gradient function."""
      with tf.control_dependencies([dy]):
        n = common_layers.layer_norm_compute(x, epsilon, norm_scale, norm_bias)
        wqkv_split = tf.unstack(wqkv, num=num_heads)
        wo_split = tf.unstack(wo, num=num_heads)
        deps = []
        dwqkvs = []
        dwos = []
        dn = 0
        for h in range(num_heads):
          with tf.control_dependencies(deps):
            combined = tf.nn.conv1d(n, wqkv_split[h], 1, 'SAME')
            q, k, v = tf.split(combined, 3, axis=2)
            o = common_attention.scaled_dot_product_attention_simple(
                q, k, v, attention_bias)
            partial_y = tf.nn.conv1d(o, wo_split[h], 1, 'SAME')
            pdn, dwqkvh, dwoh = tf.gradients(
                ys=[partial_y],
                xs=[n, wqkv_split[h], wo_split[h]],
                grad_ys=[dy])
            dn += pdn
            dwqkvs.append(dwqkvh)
            dwos.append(dwoh)
            deps = [dn, dwqkvh, dwoh]
        dwqkv = tf.stack(dwqkvs)
        dwo = tf.stack(dwos)
        with tf.control_dependencies(deps):
          dx, dnorm_scale, dnorm_bias = tf.gradients(
              ys=[n], xs=[x, norm_scale, norm_bias], grad_ys=[dn])
        return (dx, dwqkv, dwo, tf.zeros_like(attention_bias), dnorm_scale,
                dnorm_bias)

    @function.Defun(
        grad_func=grad_fn, compiled=True, separate_compiled_gradients=True)
    def forward_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
      return forward_internal(x, wqkv, wo, attention_bias, norm_scale,
                              norm_bias)

    _function_cache[key] = forward_fn

  if bias is not None:
    bias = tf.squeeze(bias, 1)
  with tf.variable_scope(name, default_name='multihead_attention', values=[x]):
    if test_vars is not None:
      wqkv, wo, norm_scale, norm_bias = list(test_vars)
    else:
      wqkv = tf.get_variable(
          'wqkv', [num_heads, 1, io_size, 3 * head_size],
          initializer=tf.random_normal_initializer(stddev=io_size**-0.5))
      wo = tf.get_variable(
          'wo', [num_heads, 1, head_size, io_size],
          initializer=tf.random_normal_initializer(
              stddev=(head_size * num_heads)**-0.5))
      norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
    y = forward_fn(x, wqkv, wo, bias, norm_scale, norm_bias)
    y.set_shape(x.get_shape())  # pytype: disable=attribute-error
    return y


class TransformerDecoder(snt.AbstractModule):
  """Transformer decoder.

  Sonnet Transformer decoder module as described in Vaswani et al. 2017. Uses
  the Tensor2Tensor multihead_attention function for masked self attention, and
  non-masked cross attention attention. Layer norm is applied inside the
  residual path as in sparse transformers (Child 2019).

  This module expects inputs to be already embedded, and does not
  add position embeddings.
  """

  def __init__(self,
               hidden_size=256,
               fc_size=1024,
               num_heads=4,
               layer_norm=True,
               num_layers=8,
               dropout_rate=0.2,
               re_zero=True,
               memory_efficient=False,
               name='transformer_decoder'):
    """Initializes TransformerDecoder.

    Args:
      hidden_size: Size of embedding vectors.
      fc_size: Size of fully connected layer.
      num_heads: Number of attention heads.
      layer_norm: If True, apply layer normalization. If mem_efficient_attention
        is True, then layer norm is always applied.
      num_layers: Number of Transformer blocks, where each block contains a
        multi-head attention layer and a MLP.
      dropout_rate: Dropout rate applied immediately after the ReLU in each
        fully-connected layer.
      re_zero: If True, alpha scale residuals with zero init.
      memory_efficient: If True, recompute gradients for memory savings.
      name: Name of variable scope
    """
    super(TransformerDecoder, self).__init__(name=name)
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.layer_norm = layer_norm
    self.fc_size = fc_size
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate
    self.re_zero = re_zero
    self.memory_efficient = memory_efficient

  def _build(self,
             inputs,
             sequential_context_embeddings=None,
             is_training=False,
             cache=None):
    """Passes inputs through Transformer decoder network.

    Args:
      inputs: Tensor of shape [batch_size, sequence_length, embed_size]. Zero
        embeddings are masked in self-attention.
      sequential_context_embeddings: Optional tensor with global context
        (e.g image embeddings) of shape
        [batch_size, context_seq_length, context_embed_size].
      is_training: If True, dropout is applied.
      cache: Optional dict containing tensors which are the results of previous
        attentions, used for fast decoding. Expects the dict to contain two
        keys ('k' and 'v'), for the initial call the values for these keys
        should be empty Tensors of the appropriate shape.
        'k' [batch_size, 0, key_channels] 'v' [batch_size, 0, value_channels]

    Returns:
      output: Tensor of shape [batch_size, sequence_length, embed_size].
    """
    if is_training:
      dropout_rate = self.dropout_rate
    else:
      dropout_rate = 0.

    # create bias to mask future elements for causal self-attention.
    seq_length = tf.shape(inputs)[1]
    decoder_self_attention_bias = common_attention.attention_bias_lower_triangle(
        seq_length)

    # If using sequential_context, identify elements with all zeros as padding,
    # and create bias to mask out padding elements in self attention.
    if sequential_context_embeddings is not None:
      pad  = common_attention.embedding_to_padding(sequential_context_embeddings)
      bias = common_attention.attention_bias_ignore_padding(pad)

    x = inputs
    for layer_num in range(self.num_layers):
      with tf.variable_scope('layer_{}'.format(layer_num)):

        # If using cached decoding, access cache for current layer, and create
        # bias that enables un-masked attention into the cache
        if cache is not None:
          layer_cache = cache[layer_num]
          layer_decoder_bias = tf.zeros([1, 1, 1, 1])
        # Otherwise use standard masked bias
        else:
          layer_cache = None
          layer_decoder_bias = decoder_self_attention_bias

        # Multihead self-attention from Tensor2Tensor.
        res = x
        if self.memory_efficient:
          res = multihead_self_attention_memory_efficient(
              res,
              bias=layer_decoder_bias,
              cache=layer_cache,
              num_heads=self.num_heads,
              head_size=self.hidden_size // self.num_heads,
              forget=True if is_training else False,
              name='self_attention'
          )
          add_activation(res, name='self_attention')
        else:
          if self.layer_norm:
            res = common_layers.layer_norm(res, name='self_attention_norm')
            add_activation(res, name='self_attention_norm')
          res = common_attention.multihead_attention(
              res,
              memory_antecedent=None,
              bias=layer_decoder_bias,
              total_key_depth=self.hidden_size,
              total_value_depth=self.hidden_size,
              output_depth=self.hidden_size,
              num_heads=self.num_heads,
              cache=layer_cache,
              dropout_rate=0.,
              make_image_summary=False,
              name='self_attention')
          add_activation(res, name='self_attention')
        if self.re_zero:
          res *= tf.get_variable('self_attention/alpha', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

        # Optional cross attention into sequential context
        if sequential_context_embeddings is not None:
          res = x
          if self.layer_norm:
            res = common_layers.layer_norm(res, name='cross_attention')
            add_activation(res, name='cross_attention_norm')
          res = common_attention.multihead_attention(
              res,
              memory_antecedent=sequential_context_embeddings,
              bias=bias,
              total_key_depth=self.hidden_size,
              total_value_depth=self.hidden_size,
              output_depth=self.hidden_size,
              num_heads=self.num_heads,
              dropout_rate=0.,
              make_image_summary=False,
              name='cross_attention')
          add_activation(res, name='cross_attention')
          if self.re_zero:
            res *= tf.get_variable('cross_attention/alpha', initializer=0.)
          if dropout_rate:
            res = tf.nn.dropout(res, rate=dropout_rate)
          x += res

        # FC layers
        res = x
        if self.layer_norm:
          res = common_layers.layer_norm(res, name='fc_norm')
          add_activation(res, name='fc_norm')
        res = tf.layers.dense(res, self.fc_size, activation=tf.nn.relu, name='fc_1')
        add_activation(res, name='fc_1')
        res = tf.layers.dense(res, self.hidden_size, name='fc_2')
        add_activation(res, name='fc_2')
        if self.re_zero:
          res *= tf.get_variable('fc/alpha', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

    if self.layer_norm:
      output = common_layers.layer_norm(x)
    else:
      output = x
    add_activation(output, name='output')
    return output

  def create_init_cache(self, batch_size):
    """Creates empty cache dictionary for use in fast decoding."""

    def compute_cache_shape_invariants(tensor):
      """Helper function to get dynamic shapes for cache tensors."""
      shape_list = tensor.shape.as_list()
      # Set shape to None for the sequence dimension
      shape_list[-2] = None
      return tf.TensorShape(shape_list)

    # Build cache
    k = common_attention.split_heads(tf.zeros([batch_size, 0, self.hidden_size]), self.num_heads)
    v = common_attention.split_heads(tf.zeros([batch_size, 0, self.hidden_size]), self.num_heads)
    cache = [{'k': k, 'v': v} for _ in range(self.num_layers)]
    shape_invariants = tf.nest.map_structure(compute_cache_shape_invariants, cache)
    return cache, shape_invariants


def conv_residual_block(inputs,
                        output_channels=None,
                        downsample=False,
                        kernel_size=3,
                        re_zero=True,
                        dropout_rate=0.,
                        name='conv_residual_block'):
  """Convolutional block with residual connections for 2D or 3D inputs.

  Args:
    inputs: Input tensor of shape [batch_size, height, width, channels] or
      [batch_size, height, width, depth, channels].
    output_channels: Number of output channels.
    downsample: If True, downsample by 1/2 in this block.
    kernel_size: Spatial size of convolutional kernels.
    re_zero: If True, alpha scale residuals with zero init.
    dropout_rate: Dropout rate applied after second ReLU in residual path.
    name: Name for variable scope.

  Returns:
    outputs: Output tensor of shape [batch_size, height, width, output_channels]
      or [batch_size, height, width, depth, output_channels].
  """
  with tf.variable_scope(name):
    input_shape = inputs.get_shape().as_list()
    num_dims = len(input_shape) - 2

    if num_dims == 2:
      conv = tf.layers.conv2d
    elif num_dims == 3:
      conv = tf.layers.conv3d

    input_channels = input_shape[-1]
    if output_channels is None:
      output_channels = input_channels
    if downsample:
      shortcut = conv(
          inputs,
          filters=output_channels,
          strides=2,
          kernel_size=kernel_size,
          padding='same',
          name='conv_shortcut')
    else:
      shortcut = inputs

    res = inputs
    res = tf.nn.relu(res)
    res = conv(
        res, filters=input_channels, kernel_size=kernel_size, padding='same',
        name='conv_1')

    res = tf.nn.relu(res)
    if dropout_rate:
      res = tf.nn.dropout(res, rate=dropout_rate)
    if downsample:
      out_strides = 2
    else:
      out_strides = 1
    res = conv(
        res,
        filters=output_channels,
        kernel_size=kernel_size,
        padding='same',
        strides=out_strides,
        name='conv_2')
    if re_zero:
      res *= tf.get_variable('alpha', initializer=0.)
  return shortcut + res


class ResNet(snt.AbstractModule):
  """ResNet architecture for 2D image or 3D voxel inputs."""

  def __init__(self,
               num_dims,
               hidden_sizes=(64, 256),
               num_blocks=(2, 2),
               dropout_rate=0.1,
               re_zero=True,
               name='res_net'):
    """Initializes ResNet.

    Args:
      num_dims: Number of spatial dimensions. 2 for images or 3 for voxels.
      hidden_sizes: Sizes of hidden layers in resnet blocks.
      num_blocks: Number of resnet blocks at each size.
      dropout_rate: Dropout rate applied immediately after the ReLU in each
        fully-connected layer.
      re_zero: If True, alpha scale residuals with zero init.
      name: Name of variable scope
    """
    super(ResNet, self).__init__(name=name)
    self.num_dims = num_dims
    self.hidden_sizes = hidden_sizes
    self.num_blocks = num_blocks
    self.dropout_rate = dropout_rate
    self.re_zero = re_zero

  def _build(self, inputs, is_training=False):
    """Passes inputs through resnet.

    Args:
      inputs: Tensor of shape [batch_size, height, width, channels] or
        [batch_size, height, width, depth, channels].
      is_training: If True, dropout is applied.

    Returns:
      output: Tensor of shape [batch_size, height, width, output_size] or
        [batch_size, height, width, depth, output_size] where output_size ==
        self.hidden_sizes[-1].
    """
    add_activation(inputs, name='inputs')

    dropout_rate = self.dropout_rate if is_training else 0.0

    # Initial projection with large kernel as in original resnet architecture
    if self.num_dims == 3:
      conv = tf.layers.conv3d
    elif self.num_dims == 2:
      conv = tf.layers.conv2d
    x = conv(
        inputs,
        filters=self.hidden_sizes[0],
        kernel_size=7,
        strides=2,
        padding='same',
        name='conv_input')
    add_activation(x, name='conv_input')

    if self.num_dims == 2:
      x = tf.layers.max_pooling2d(
          x, strides=2, pool_size=3, padding='same', name='pool_input')
      add_activation(x, name='pool_input')

    for d, (hidden_size, blocks) in enumerate(zip(self.hidden_sizes, self.num_blocks)):
      with tf.variable_scope('resolution_{}'.format(d)):
        for i in range(blocks):
          # Downsample at the start of each collection of blocks
          x = conv_residual_block(
              x,
              downsample=(i == 0 and d != 0),
              dropout_rate=dropout_rate,
              output_channels=hidden_size,
              re_zero=self.re_zero,
              name=f'block_{i}')
          add_activation(x, name=f'block_{i}')

    return x


class MLPDecoder(snt.AbstractModule):
  def __init__(self, hidden_size=512, fc_size=256, num_layers=4,
               layer_norm=True, re_zero=True, dropout_rate=0.01, width=201,
               name='mlp_decoder', **kwds):
    """MLP decoder (not My Little Pony)

      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between linear layers. Defaults
        to ReLU.
      dropout_rate: Dropout rate to apply, a rate of `None` (the default) or `0`
        means no dropout will be applied.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.
    """
    if kwds:
        warnings.warn(f'ignoring keywords: {kwds}')
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.fc_size = fc_size
    self.num_layers = num_layers
    self.layer_norm = layer_norm
    self.re_zero = re_zero
    self.dropout_rate = dropout_rate
    self.width = width

  def _windows(self, inputs):
      batch_size = tf.shape(inputs)[0]

      sos_embed = tf.get_variable('embed_sos', shape=[1, 1, self.hidden_size],
                                  initializer=tf.glorot_uniform_initializer)

      sos_embed_tiled = tf.tile(sos_embed, [batch_size, self.width - 1, 1])

      # inputs is [batch_size, seq_length, E], prepend "start of sequence"
      # embeddings thus [batch_size, seq_length + width - 1, E].
      padded = tf.concat((sos_embed_tiled, inputs), axis=1)

      # Slice inputs so we get `width` offsetted slices.
      slices = []
      for start in range(self.width):
        stop = None if start - self.width + 1 == 0 else start - self.width + 1
        idx = [slice(None), slice(start, stop)]
        slices.append(padded[idx])

      return tf.stack(slices, axis=2)

  def _build(self,
             inputs,
             sequential_context_embeddings=None,
             is_training=False,
             cache=None):

    dropout_rate = self.dropout_rate if is_training else 0.

    width = self.width
    input_shape = tf.shape(inputs)
    batch_size, seq_length = input_shape[0], input_shape[1]

    windows = self._windows(inputs)

    x = tf.reshape(windows, [batch_size, seq_length, width*self.hidden_size])

    for layer_num in range(self.num_layers):
      with tf.variable_scope('layer_{}'.format(layer_num)):
        res = x
        if self.layer_norm:
          res = common_layers.layer_norm(res, name='norm')
          add_activation(res, name='norm')
        res = tf.layers.dense(res, width*self.fc_size, activation=tf.nn.relu, name='fc')
        add_activation(res, name='fc')
        res = tf.reshape(res, [batch_size, seq_length, width, self.fc_size])
        res = tf.layers.dense(res, self.hidden_size, name='project')
        res = tf.reshape(res, [batch_size, seq_length, width*self.hidden_size])
        add_activation(res, name='project')
        if self.re_zero:
          res *= tf.get_variable('fc/alpha', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

    with tf.variable_scope('output'):
      if self.layer_norm:
        x = common_layers.layer_norm(x, name='norm_end')

    add_activation(x, name='output')
    return x

  def create_init_cache(self, num_samples):
      return (), ()


class MLPMixerDecoder(MLPDecoder):
  def __init__(self, hidden_size=512, fc_size=256, num_layers=4,
               layer_norm=True, re_zero=True, dropout_rate=0.01, width=201,
               name='mlp_decoder', **kwds):
    if kwds:
        warnings.warn(f'ignoring keywords: {kwds}')
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.fc_size = fc_size
    self.num_layers = num_layers
    self.layer_norm = layer_norm
    self.re_zero = re_zero
    self.dropout_rate = dropout_rate
    self.width = width

  def _build(self,
             inputs,
             sequential_context_embeddings=None,
             is_training=False,
             cache=None):

    dropout_rate = self.dropout_rate if is_training else 0.

    x = self._windows(inputs)

    for layer_num in range(self.num_layers):
      with tf.variable_scope('layer_{}'.format(layer_num)):

        res = x
        if self.layer_norm:
          res = common_layers.layer_norm(res, name='norm1')
        res = tf.transpose(res, (0, 1, 3, 2))
        res = tf.layers.dense(res, self.fc_size, activation=tf.nn.relu, name='fc1a')
        res = tf.layers.dense(res, self.width, name='fc1b')
        res = tf.transpose(res, (0, 1, 3, 2))
        if self.re_zero:
          res *= tf.get_variable('fc/alpha1', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

        res = x
        if self.layer_norm:
          res = common_layers.layer_norm(res, name='norm2')
        res = tf.layers.dense(res, self.fc_size, activation=tf.nn.relu, name='fc2a')
        res = tf.layers.dense(res, self.hidden_size, name='fc2b')
        if self.re_zero:
          res *= tf.get_variable('fc/alpha2', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

    with tf.variable_scope('output'):
      if self.layer_norm:
        x = common_layers.layer_norm(x, name='norm_end')

    x = tf.reduce_mean(x, axis=-2)

    return x

  def create_init_cache(self, num_samples):
      return (), ()


class PolyLineModel(snt.AbstractModule):

  def __init__(self, *,
               decoder_config: dict,
               tuplet_size: int = 3,
               quantization_bits: int,
               use_discrete_embeddings: bool = True,
               max_seq_length: int = 5000,
               max_num_input_triplets: int = 2500,
               use_position_embeddings: bool = True,
               name: str = 'polyline_model'):
    super(PolyLineModel, self).__init__(name=name)
    self.tuplet_size             = tuplet_size
    self.embedding_dim           = decoder_config['hidden_size']
    self.max_seq_length          = max_seq_length
    self.quantization_bits       = quantization_bits
    self.max_num_input_triplets  = max_num_input_triplets
    self.use_discrete_embeddings = use_discrete_embeddings
    self.use_position_embeddings = use_position_embeddings

    with self._enter_variable_scope():
      decoder_config = dict(decoder_config)
      decoder_model = decoder_config.pop('model', 'transformer')
      if decoder_model == 'transformer':
        self.decoder = TransformerDecoder(**decoder_config)
      elif decoder_model == 'mlp':
        self.decoder = MLPDecoder(**decoder_config)
      elif decoder_model == 'mlpmixer':
        self.decoder = MLPMixerDecoder(**decoder_config)
      else:
        raise ValueError(f'{decoder_model!r} is not a valid model name')

  @property
  def vocab_size(self):
    return 2**self.quantization_bits + len(code_tokens)

  @snt.reuse_variables
  def _prepare_context(self, context, is_training=False):
    global_context_embedding = None
    sequential_context_embeddings = None
    return global_context_embedding, sequential_context_embeddings

  @snt.reuse_variables
  def _embed_inputs(self, tokens, global_context_embedding=None):
    """Embeds flat vertices and adds position and coordinate information."""

    # Cut sequence to maximum input length
    #tokens = tokens[:, :self.tuplet_size*self.max_num_input_triplets]

    # Dequantize inputs and get shapes
    input_shape = tf.shape(tokens)
    batch_size, seq_length = input_shape[0], input_shape[1]

    # E(t_i) = E_coord(t_i) + E_trip(t_i) + E_token(t_i)
    # E_coord(t_i) : N x N -> R^{embed_dim}
    # E_trip(t_i)  : N x N -> R^{embed_dim}
    # E_token(t_i) : N x N -> R^{embed_dim}
    # E(t_i)       : N x N -> R^{embed_dim}

    # Our representation:
    # t = (code0, x0, y0,
    #      code1, x1, y1,
    #      …,
    #      codeN, xN, yN)    seq_length := N

    # Dimension indicator embeddings (code, x, y)
    coord_embeddings = snt.Embed(vocab_size=self.tuplet_size,
                                 embed_dim=self.embedding_dim,
                                 initializers={'embeddings': tf.glorot_uniform_initializer},
                                 densify_gradients=True,
                                 name='coord_embeddings')(tf.mod(tf.range(seq_length), self.tuplet_size))

    # Position embeddings
    pos_embeddings = snt.Embed(vocab_size=self.max_num_input_triplets,
                               embed_dim=self.embedding_dim,
                               initializers={'embeddings': tf.glorot_uniform_initializer},
                               densify_gradients=True,
                               name='pos_embeddings')(tf.floordiv(tf.range(seq_length), self.tuplet_size))

    # Continuous token value embeddings
    if not self.use_discrete_embeddings:
      raise NotImplementedError

    # Discrete token value embeddings
    token_embeddings = snt.Embed(vocab_size=self.vocab_size,
                                 embed_dim=self.embedding_dim,
                                 initializers={'embeddings': tf.glorot_uniform_initializer},
                                 densify_gradients=True,
                                 name='token_embeddings')(tokens)

    # Step zero embeddings
    if global_context_embedding is None:
      zero_embed = tf.get_variable('embed_zero', shape=[1, 1, self.embedding_dim])
      zero_embed_tiled = tf.tile(zero_embed, [batch_size, 1, 1])
    else:
      zero_embed_tiled = global_context_embedding[:, None]

    # Aggregate embeddings
    annotations = coord_embeddings
    if self.use_position_embeddings:
      annotations += pos_embeddings
    embeddings = token_embeddings + annotations[None]
    embeddings = tf.concat([zero_embed_tiled, embeddings], axis=1)

    return embeddings

  @snt.reuse_variables
  def _project_to_logits(self, inputs):
    """Projects transformer outputs to logits for predictive distribution."""
    return tf.layers.dense(inputs,
                           self.vocab_size,
                           use_bias=True,
                           kernel_initializer=tf.zeros_initializer(),
                           name='project_to_logits')

  @snt.reuse_variables
  def _create_dist(self,
                   tokens,
                   global_context_embedding=None,
                   sequential_context_embeddings=None,
                   temperature=1.,
                   top_k=0,
                   top_p=1.,
                   is_training=False,
                   return_whole=False,
                   cache=None):
    """CPD over next token value."""

    # Embed inputs
    inputs = self._embed_inputs(tokens, global_context_embedding)

    # The cache contains the key and queries for indices until the last one,
    # which is the new one. Therefore, if we have a cache, only query for the
    # last element.
    if cache:
      if return_whole:
        raise ValueError('cannot return whole sequence with caching enabled')
      inputs = inputs[:, -1:]

    # inputs.shape  = [batch_size, seq_len, embedding_dim], and
    # outputs.shape = [batch_size, seq_len, output_dim].
    # As previously noted, seq_len is 1 if cache is not None.
    outputs = self.decoder(inputs, cache=cache,
                           sequential_context_embeddings=sequential_context_embeddings,
                           is_training=is_training)

    # If there is no cache, the decoder returns the entire sequence. Therefore,
    # we cut anything before the last one /unless otherwise requested./
    if not cache and not return_whole:
      outputs = outputs[:, -1:, :]

    # Get logits and optionally process for sampling
    logits = self._project_to_logits(outputs)
    logits /= temperature

    logits = top_k_logits(logits, top_k)
    logits = top_p_logits(logits, top_p)
    add_activation(logits, name='logits')

    return tfd.Categorical(logits=logits)

  def _build(self, batch, is_training=False):
    global_context, seq_context = self._prepare_context(batch, is_training=is_training)
    pred_dist = self._create_dist(batch['tokens'][:, :-1],
                                  global_context_embedding=global_context,
                                  sequential_context_embeddings=seq_context,
                                  is_training=is_training,
                                  return_whole=True)
    return pred_dist

  def sample(self,
             num_samples=None,
             context=None,
             max_num_triplets=None,
             #min_num_triplets=0,
             temperature=1.,
             top_k=0,
             top_p=1.,
             prompts=None,
             recenter_verts=True,
             only_return_complete=True):
    # Obtain context for decoder
    global_context, seq_context = self._prepare_context(context, is_training=False)

    if prompts is None and num_samples is not None:
      samples = tf.zeros([num_samples, 0], dtype=tf.int32)
    elif num_samples is None and prompts is not None:
      samples = tf.identity(prompts)
      num_samples = tf.shape(prompts)[0]
    else:
      raise ValueError('must specify num_samples or prompts, but not both')

    # num_samples is the minimum value of num_samples and the batch size of
    # context inputs (if present).
    if global_context is not None:
      num_samples = tf.minimum(num_samples, tf.shape(global_context)[0])
      global_context = global_context[:num_samples]
      if seq_context is not None:
        seq_context = seq_context[:num_samples]
    elif seq_context is not None:
      # We don't want to do
      #   num_samples = tf.minimum(num_samples, tf.shape(seq_context)[0])
      # because if num_samples is a simple Python integer, the above would then
      # be a TF tensor which becomes a problem later as we need to know that
      # dimension.
      seq_context = seq_context[:num_samples]

    def body(i, samples, cache=None):
      cat_dist = self._create_dist(
          samples,
          global_context_embedding=global_context,
          sequential_context_embeddings=seq_context,
          cache=cache,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p)
      next_sample = cat_dist.sample()
      samples = tf.concat([samples, next_sample], axis=1)
      return (i+1, samples) if cache is None else (i+1, samples, cache)

    def cond(i, samples, cache=None):
      del i, cache  # Unused
      # Continue if there is any sample that contains no stop token.
      not_eos = tf.not_equal(samples, sequence_end_token)
      incomplete_seqs = tf.reduce_all(not_eos, axis=-1)
      return tf.reduce_any(incomplete_seqs)

    # Set up the loop
    max_num_triplets = max_num_triplets or self.max_num_input_triplets
    max_iters = max_num_triplets * 3 + 1 - tf.shape(samples)[1]
    loop_vars = (0, samples)
    shape_invars = (tf.TensorShape([]),
                    tf.TensorShape([None, None]))

    # Only use cache when starting from scratch.
    if prompts is None:
      cache, cache_shape_invars = self.decoder.create_init_cache(num_samples)
      loop_vars    = (*loop_vars,    cache)
      shape_invars = (*shape_invars, cache_shape_invars)

    # After the while loop, tokens.shape = [num_samples, sample_length] where
    #   sample_length < 3*max_num_triplets + 1.
    tokens = tf.while_loop(cond=cond,
                           body=body,
                           loop_vars=loop_vars,
                           shape_invariants=shape_invars,
                           maximum_iterations=max_iters,
                           back_prop=False,
                           parallel_iterations=1)[1]

    # Check if samples completed. Samples are complete if the stopping token
    # is produced.
    completed = tf.reduce_any(tf.equal(tokens, sequence_end_token), axis=-1)

    # Get the number of triplets in the sample. This requires finding the
    # index of the stopping token. For complete samples use to argmax to get
    # first nonzero index.
    stop_index_completed = tf.argmax(
        tf.cast(tf.equal(tokens, sequence_end_token), tf.int32), axis=-1, output_type=tf.int32)
    # For incomplete samples the stopping index is just the maximum index.
    stop_index_incomplete = 3*max_num_triplets*tf.ones_like(stop_index_completed)
    stop_index = tf.where(completed, stop_index_completed, stop_index_incomplete)

    # Number of triplets in each sampled sequence
    num_sampled = tf.floordiv(stop_index, 3)

    # Number of triplets in longest sampled sequence
    max_num_sampled = tf.reduce_max(num_sampled)

    # Cut sequences at longest sampled sequence. Realistically only cuts a few
    # tokens since sampling only stops when all samples have a stop token.
    tokens = tokens[:, :3*max_num_sampled]

    triplets = tf.reshape(tokens, [num_samples, max_num_sampled, 3])

    # Pad samples to max sample length. Pad with stopping tokens for incomplete
    # samples. After this is done,
    #   triplets.shape = [num_samples, max_num_triplets, 3].
    pad_size = max_num_triplets - max_num_sampled
    assert sequence_end_token == 0
    triplets = tf.pad(triplets, [[0, 0], [0, pad_size], [0, 0]])

    # Triplets mask
    triplets_mask = tf.range(max_num_triplets)[None] < num_sampled[:, None]
    triplets_mask = tf.cast(triplets_mask, tf.int32)

    codes    = triplets[..., 0]
    vertices = triplets[..., 1:]

    # Adjust for the three opcodes (moveto/lineto/EOS)
    vertices -= len(code_tokens)

    if recenter_verts:
      raise NotImplementedError('needs to be adapted for quantized')
      #large_number_mask = 0xffff * (1 - triplets_mask)[..., None]
      #vert_max = tf.reduce_max(vertices - large_number_mask, axis=1, keepdims=True)
      #vert_min = tf.reduce_min(vertices + large_number_mask, axis=1, keepdims=True)
      #vert_centers = (vert_max + vert_min)/2.0
      #vertices -= vert_centers

    triplets = tf.stack([codes,
                         vertices[..., 0],
                         vertices[..., 1]], axis=-1)

    # Zeros out anything beyond the end of a sequence
    triplets *= triplets_mask[..., None]

    if only_return_complete:
      triplets = tf.boolean_mask(triplets, completed)
      num_sampled = tf.boolean_mask(num_sampled, completed) + 1
      triplets_mask = tf.boolean_mask(triplets_mask, completed)
      completed = tf.boolean_mask(completed, completed)

    # Outputs
    outputs = {
        'completed': completed,
        'triplets': triplets,
        'num_triplets': num_sampled,
        'triplets_mask': triplets_mask,
    }
    return outputs


class ImageToPolyLineModel(PolyLineModel):
  def __init__(self, *, image_encoder, name='image_to_polyline_model', **kwargs):
    """
    Args:
      See PolyLineModel, then
      image_encoder: Image encoder
      name: Name of variable scope
    """
    super(ImageToPolyLineModel, self).__init__(**kwargs)
    self.image_encoder = image_encoder

  @snt.reuse_variables
  def _prepare_context(self, context, is_training=False):

    # Pass images through encoder
    # The image tensor shape is
    #   context['image'].shape = [batch_size, height, width, channels].
    image = context['image'] - 0.5
    with tf.control_dependencies([tf.assert_less_equal(image, 0.5),
                                  tf.assert_greater_equal(image, -0.5)]):
      image_embeddings = self.image_encoder(image, is_training=is_training)
    # Output shape is
    #   image_embeddings.shape = [batch_size, height, width, embedding_dim].

    # Add 2D coordinate grid embedding
    batch_size = tf.shape(image_embeddings)[0]
    height = image_embeddings.shape[1]
    width = image_embeddings.shape[2]
    y = tf.linspace(-1., 1., height)
    x = tf.linspace(-1., 1., width)
    image_coords = tf.stack(tf.meshgrid(x, y), axis=-1)
    # image_coords.shape = [height, width, 2]
    image_coord_embeddings = tf.layers.dense(
        image_coords,
        self.embedding_dim,
        use_bias=True,
        name='image_coord_embeddings')
    # W.shape = [2, embedding_dim]
    # image_coord_embeddings.shape       =    [height, width, embedding_dim]
    # image_coord_embeddings[None].shape = [1, height, width, embedding_dim]
    image_embeddings += image_coord_embeddings[None]

    # Reshape spatial grid to sequence
    sequential_context_embedding = tf.reshape(
        image_embeddings, [batch_size, height*width, self.embedding_dim])

    # Result is of the shape [batch_size, height*width, embedding_dim].
    return None, sequential_context_embedding


def mlp_block(x, dim):
  y = tf.layers.dense(x, dim, activation=tf.nn.relu)
  y = tf.layers.dense(y, x.shape[-1])
  return y


def mlp_mixer_block(x, tokens_dim, channels_dim):
  with tf.variable_scope('token_mix'):
    y = common_layers.layer_norm(x)
    y = tf.transpose(y, (0, 2, 1))
    y = mlp_block(y, tokens_dim)
    y = tf.transpose(y, (0, 2, 1))
    x += y
  with tf.variable_scope('channel_mix'):
    y = common_layers.layer_norm(x)
    y = mlp_block(y, channels_dim)
    x += y
  return x


class MLPMixerImageEncoder(snt.AbstractModule):
  def __init__(self, *, num_dims=2, re_zero=True, dropout_rate=0.01,
               num_layers, patch_size, hidden_size, tokens_mlp_size,
               channels_mlp_size, name='mlp_mixer'):
    super().__init__(name=name)
    self.num_dims = num_dims
    self.re_zero = re_zero
    self.dropout_rate = dropout_rate
    self.num_layers = num_layers
    self.patch_size = patch_size
    self.hidden_size = hidden_size
    self.tokens_mlp_size = tokens_mlp_size
    self.channels_mlp_size = channels_mlp_size

  def _build(self, inputs, is_training=False):

    dropout_rate = self.dropout_rate if is_training else 0.

    if self.num_dims == 3:
      conv = tf.layers.conv3d
    elif self.num_dims == 2:
      conv = tf.layers.conv2d

    s = self.patch_size
    h = self.hidden_size
    x = conv(inputs, filters=h, kernel_size=s, strides=s,
             padding='same', name='stem')

    batch_size = tf.shape(x)[0]
    input_size = x.shape[1:1+self.num_dims]
    x = tf.reshape(x, [batch_size, np.prod(input_size), h])

    for layer_num in range(self.num_layers):
      with tf.variable_scope(f'mixer_{layer_num}'):
        res = x
        res = mlp_mixer_block(res, self.tokens_mlp_size, self.channels_mlp_size)
        if self.re_zero:
          res *= tf.get_variable('alpha', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

    x = common_layers.layer_norm(x, name='norm')
    x = tf.reshape(x, [batch_size, *input_size, h])
    return x
