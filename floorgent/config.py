import os
import sys
import inspect
import argparse
import warnings
import importlib
from time import sleep
from pprint import pformat

import tensorflow.compat.v1 as tf
from .utils import Tee, implies


if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
    m = 'CUDA_VISIBLE_DEVICES is unset. Remove this warning and delay by setting it.'
    warnings.warn(m)
    sleep(5.0)


tf_config = tf.ConfigProto()


configurable_names = r"""
conditioning            COND Type of conditioning to use (image or none).
tuplet_size             N    Tuplet size (triplet, sextet, etc).
quant_bits              N    Quantization bits
max_num_points          N    Number of positions to pick sampling locations from.
max_num_segs            N    Number of nearest segments to keep for each generated sample.
train_pct               PCT  Training/test split percentage \in [0, 1].
pos_embeddings          Y/N  Whether to include position embeddings.
decoder_model           MDL  Decoder model specifier.
decoder_hidden_size     N    Attention layer hidden units.
decoder_fc_size         N    Fully-connected layer hidden units.
decoder_num_heads       N    Number of heads.
decoder_num_layers      N    Number of layers.
decoder_dropout_rate    PCT  Dropout rate \in [0, 1] for Transformer decoder.
image_encoder_model     MDL  Image encoder model specifier.
resnet_dropout_rate     PCT  Dropout rate \in [0, 1] for ResNet input layers.
image_format            DIM  Image format specification.
max_bev_segs            N    Limit number of segments to pass to BEV rasterizer.
batch_size              N    Batch size in samples.
learning_rate           ETA  Gradient step size.
training_steps          N    Number of training steps until termination.
plot_step               N    Plot output of network every N steps.
loss_step               N    Store loss summary every N steps.
save_step               N    Save network weights every N steps.
out_dir                 DIR  Store model in DIR and load if possible.
comment                 TEXT Annotate this run with TEXT.
dataset_spec            DS   Dataset specification
""".strip().splitlines()


def boolish(val):
    if val.casefold() in {'yes', 'y', '有'}:
        return True
    elif val.casefold() in {'no', 'n', '不'}:
        return False
    else:
        raise ValueError(f'Expected yes/y or no/n, got {val!r}')



def _make_argparser():
    formatter_class = lambda **k: argparse.ArgumentDefaultsHelpFormatter(max_help_position=100, **k)
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    for line in configurable_names:
        name, metavar, desc = line.strip().split(maxsplit=2)
        default = getattr(config, name)
        opt = f'--{name.replace("_", "-")}'
        if name == 'dataset_spec':
            parser.add_argument(dest=name, nargs='?', help=desc, default=default)
        else:
            typ = type(default)
            typ = boolish if typ is bool else typ
            parser.add_argument(opt, dest=name, help=desc, type=typ,
                                metavar=metavar, default=default)
    return parser


def update_from_args():
    args = parse_args()
    config_dict = dict(args.__dict__)
    update(**config_dict)


def update(**change):

    for old, new in [('num_segments',  'max_num_segs'),
                     ('num_positions', 'max_num_points'),
                     ('min_dist',      'min_point_dist')]:
        if old in change:
            warnings.warn(f'remapping old config key {old!r} to {new!r}')
            assert new not in change
            change[new] = change.pop(old)

    unused = set(change) - set(config.__dict__)
    if unused:
        warnings.warn(f'update() with unused keys {unused}, this is likely a mistake')

    config.__dict__.update(change)
    _set_computed_configs()


def _set_computed_configs():
    "Ensure consistency in config"

    config.quant_max     = 2**config.quant_bits - 1
    config.quant_range   = config.quant_max - config.quant_min
    config.dequant_min   = -config.dequant_range/2
    config.dequant_max   = +config.dequant_range/2

    config.pos_embeddings = bool(int(config.pos_embeddings))

    config.image_shape = tuple(map(int, config.image_format.split('x')))

    tf_config.gpu_options.allow_growth = config.tf_gpu_options_allow_growth

    if config.conditioning == 'bev':
        (h, w, c) = config.image_shape
        if h != w:
            warnings.warn('non-square image in bev, setting square')
            d = max(h, w)
            config.image_format = f'{d}x{d}x{c}'
            config.image_shape = (d, d, c)

    assert implies(config.conditioning != 'none',
                   config.resnet_hidden_sizes[-1] == config.decoder_hidden_size)

    config.model_base_config = dict(
        decoder_config={**config.decoder_opts,
                        'model':        config.decoder_model,
                        'hidden_size':  config.decoder_hidden_size,
                        'fc_size':      config.decoder_fc_size,
                        'num_heads':    config.decoder_num_heads,
                        'num_layers':   config.decoder_num_layers,
                        'dropout_rate': config.decoder_dropout_rate},
        tuplet_size=config.tuplet_size,
        max_num_input_triplets=2*config.max_num_segs + 1,
        quantization_bits=config.quant_bits,
        use_position_embeddings=config.pos_embeddings)

    config.resnet_config = {**config.resnet_opts,
                            'hidden_sizes': config.resnet_hidden_sizes,
                            'num_blocks':   config.resnet_num_blocks,
                            'dropout_rate': config.resnet_dropout_rate}


def dump(*, save_file=False):
    if save_file:
        config.log_dir.mkdir(parents=True, exist_ok=False)
        with open(config.log_dir / save_file, 'w') as save_file_f:
            _dump_into(Tee(sys.stdout, save_file_f))
    else:
        _dump_into(sys.stdout)


def _dump_into(f):
    print('# config module attributes:', file=f)
    for name, value in config.__dict__.items():
        if name.startswith('_') or name in _ignore_attrs:
            continue
        elif hasattr(value, '__code__'):
            print(inspect.getsource(value), end='', file=f)
        else:
            print(f'{name} = {pformat(value)}', file=f)


config_name = '.configs.base'
if os.environ.get('TEST'):
    config_name = '.configs.quicktest'
config = importlib.import_module(config_name, package=__package__)


# Switcheroo
_self_mod = sys.modules[__name__]
_ignore_attrs = set(_self_mod.__dict__)
_self_mod.__dict__.update({k: v for k, v in config.__dict__.items()
                           if not (k.startswith('_') or k in _ignore_attrs)})
config = _self_mod
_set_computed_configs()


argparser = _make_argparser()
parse_args = argparser.parse_args
_ignore_attrs |= {'argparser', 'parse_args'}


# This hack just helps the autocomplete find the configuration variable names.
if 0:
    from configs.base import *  # noqa
