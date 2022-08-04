import datetime
from pathlib import Path

# Triplets, sextets, etc
tuplet_size  = 3

# Quantization
quant_bits = 8
quant_max = 2**quant_bits - 1
quant_min = 0
quant_range = quant_max - quant_min
# Dequantized representation (aka unit cube normalized)
dequant_range = 1.0
dequant_min   = -dequant_range/2
dequant_max   = +dequant_range/2

# Number of positions to pick sampling points from.
max_num_points = 1000

# Minimum distance between sampled locations.
min_point_dist = 0.5

# Truncation distance in meters.
truncation_dist = 0.20

# Maximum/minimum number of segments per sample.
max_num_segs = 100
min_num_segs =  10

# Line soup representation.
line_soup = True

# Maximum segment length before breaking in half.
max_seg_len = 2.5

# Maximum segment distance from sampled point. Effectively a radius.
max_seg_dist = 7.5

# Minimum segment distance from sampled point.
min_seg_dist = 0.4

# Training/test split percentage \in [0, 1].
train_pct = 0.9

# Don't eat all GPU memory at once, allocate as you go.
tf_gpu_options_allow_growth = True

# Conditioning
conditioning = 'none'

# Batch size
batch_size = 8

# Embedding dimension
E = 512

# Transformer decoder configuration
decoder_model        = 'transformer'
decoder_hidden_size  = E
decoder_fc_size      = E
decoder_num_heads    = 8
decoder_num_layers   = 6
decoder_dropout_rate = 0.6
decoder_opts = {'layer_norm': True,
                're_zero': True,
                'memory_efficient': True}

# Actual input is 1024x2048x3 but stride by 4
image_format = '256x512x3'
image_stride = 4

# Limit number of segments to pass to BEV rasterizer.
max_bev_segs = 999

# Default to ResNet
image_encoder_model = 'resnet'

# Each ResNet layer downsamples by half on each side length.
resnet_hidden_sizes = (16,  E)
resnet_num_blocks =   ( 2,  4)
resnet_dropout_rate = 0.1
resnet_opts = {'re_zero': True,
               'name': 'polyline_model/res_net'}


# MLP Mixer image encoder configuration
mlpmix_encoder_config = {'re_zero': True,
                         'dropout_rate': 0.1,
                         'num_layers': 6,
                         'patch_size': 8,
                         'hidden_size': E,
                         'tokens_mlp_size': 256,
                         'channels_mlp_size': 256}

# Jittering for translations. Not possible with position-dependent
# conditionings such as camera images; we cannot translate them.
shift_factor = 0.25

# The jittering angles for panoramas are sampled from a mean-zero normal
# distribution and is truncated at 2-sigma by resampling, creating a
# lower/upper bound.
pitch_roll_stddev_deg = 2.5/2
yaw_stddev_deg = 0.0  #180.0/2

# If true, add a sequence position embedding to each token.
pos_embeddings = True

# Number of sample outputs during training.
num_sample_outputs = 4

debug = False
learning_rate  = 3e-4
training_steps = 2_000_000
plot_step = 10_000
loss_step = 100
save_step = 10_000

# Input directory
kth_dataset_dir = Path('/ds/kth_floorplans')
dataset_spec = f'kth:{kth_dataset_dir}/*/*.xml'

# Output directories
out_dir      = Path('runs') / f'{datetime.datetime.now():%Y%m%d-%H%M%S}'
model_dir    = out_dir / 'model'
log_dir      = out_dir

# Comment set by user
comment = ''
