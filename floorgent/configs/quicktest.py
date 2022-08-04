from configs.base import *

dataset_spec = f'kth:{kth_dataset_dir}/A0043022/*.xml'

debug = True

decoder_hidden_size  = 128
decoder_fc_size      = 128
decoder_num_heads    = 4
decoder_num_layers   = 3

resnet_hidden_sizes = (*resnet_hidden_sizes[:-1], decoder_hidden_size)

learning_rate = 1e-3
training_steps = 2_000
plot_step = 10
save_step = 10
summary_step = 1
