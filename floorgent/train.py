#!/usr/bin/env python
"Floor plans and transformers"

import os
import logging
import datetime
import warnings

import logcolor

if 1:
    # Must set up logging before imports
    logcolor.basic_config()

if __name__ == '__main__':
    print('Importing matplotlib + numpy...', end='', flush=True)

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print('\x1b[2K\r', end='', flush=True)
    print('Importing tensorflow...', end='', flush=True)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'^Passing .* as a synonym')
    import tensorflow._api.v1.compat.v1 as tf
    tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages

if __name__ == '__main__':
    print('\x1b[2K\r', end='', flush=True)
    print('Importing FloorGenT...', end='', flush=True)

from . import stats
from . import config

if __name__ == '__main__':
    config.update_from_args()

from . import datasets
from .utils import statstr
from .utils import split_pct
from .modules import ResNet
from .modules import PolyLineModel
from .modules import ImageToPolyLineModel
from .modules import MLPMixerImageEncoder
from .polylines import plot_path
from .polylines import ConversionError
from .polylines import triplets_to_path
from .polylines import figure_to_ndarray
from .polylines_tf import get_batch
from .polylines_tf import batch_loss
from .polylines_tf import dataset_from_ex_list


if __name__ == '__main__':
    print('\x1b[2K\r', end='', flush=True)


log = logging.getLogger('train')


def plot_sample(sample_path=None, true_path=None, ax=None, margin_pct=10):
    ax = plt.gca() if ax is None else ax
    verts_list = []

    if true_path is not None:
        plot_path(true_path, label='Actual', color='C3', text_x=0.24, ax=ax)
        verts_list.append(true_path.vertices)

    if sample_path is not None:
        plot_path(sample_path, label='Sampled', color='C0', text_x=0.04, ax=ax)
        verts_list.append(sample_path.vertices)

    # Mark origin
    ax.scatter(0.0, 0.0, marker='*', s=100, color='C5')

    # Compute appropriate bounding box
    verts = np.concatenate(verts_list, axis=0)
    bbox = np.array([verts.min(axis=0),
                     verts.max(axis=0)])
    center = bbox.mean(axis=0)
    sidelen = (1 + margin_pct/100)*np.max(np.diff(bbox, axis=0))
    bbox = np.array([center - sidelen/2,
                     center + sidelen/2])
    xlim = bbox[:, 0]
    ylim = bbox[:, 1]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_box_aspect(1.0)


def make_model():
    if config.conditioning in {'pano', 'bev'}:
        if config.image_encoder_model == 'resnet':
            image_encoder = ResNet(num_dims=2, **config.resnet_config)
        if config.image_encoder_model == 'mlpmixer':
            image_encoder = MLPMixerImageEncoder(num_dims=2, **config.mlpmix_encoder_config)
        model = ImageToPolyLineModel(**config.model_base_config,
                                     image_encoder=image_encoder)
    elif config.conditioning == 'none':
        model = PolyLineModel(**config.model_base_config)
    else:
        0/0

    print('Instantiated model under scope', model.scope_name)
    assert not model.scope_name[-1].isdigit(), 'Model must have only one instance'

    return model


def main():

    config.dump(save_file='config.py')
    with open(config.out_dir / 'pid', 'w') as f:
        f.write(f'{os.getpid()}\n')

    print()
    print()

    ################
    # Data loading #
    ################

    print(f'Loading dataset {config.dataset_spec}')
    dataset = datasets.load(config.dataset_spec)
    print()

    print(f'Generating examples from {dataset.size}')
    print()

    t0 = datetime.datetime.now()
    ex_list = dataset.generate_examples()
    t1 = datetime.datetime.now()

    print('Generation finished')
    print('#examples:', len(ex_list))
    print('triplets/ex:', statstr([len(ex.triplets) for ex in ex_list]))

    print(f'Elapsed time: {(t1-t0).total_seconds():.2f}s')
    print(f'Dataset center: {dataset.center}')
    print(f'Dataset  scale: {dataset.scale}')

    #stats.print_summary()
    #0/0

    train_ex_list, val_ex_list = split_pct(ex_list, pct=config.train_pct)

    # If we shuffle *after* the test/training split, we increase separation of
    # training and validation somewhat - a given building will tend to be in only
    # one of the two, not mixed between both as would be the case otherwise.
    np.random.shuffle(train_ex_list)
    np.random.shuffle(val_ex_list)

    print(f'{len(train_ex_list)} training examples')
    print(f'{len(val_ex_list)} validation examples')

    print()
    print()

    ##################################
    # Building the computation graph #
    ##################################


    with tf.variable_scope('dataset'):
        train_ds = dataset_from_ex_list(train_ex_list)
        val_ds   = dataset_from_ex_list(val_ex_list)

    model = make_model()

    with tf.variable_scope('train'):
        train_batch   = get_batch(train_ds, batch_size=config.batch_size)
        train_context = batch_loss(model, train_batch, is_training=True)

    with tf.variable_scope('val'):
        val_batch   = get_batch(val_ds, batch_size=config.batch_size)
        val_context = batch_loss(model, val_batch, is_training=False)

    glob_context, seq_context = model._prepare_context(train_context)
    print(f'    Global context: {getattr(glob_context, "shape", None)}')
    print(f'Sequential context: {getattr(seq_context, "shape", None)}')
    print()
    del glob_context, seq_context


    if config.plot_step:
      with tf.variable_scope('train_sample'):
        train_samples = model.sample(
            config.num_sample_outputs, context=train_context, max_num_triplets=200,
            top_p=0.80, recenter_verts=False, only_return_complete=False)


    with tf.variable_scope('optimize'):
        learning_rate = tf.placeholder(tf.float64, shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optim_op  = optimizer.minimize(train_context['loss'])


    histogram_summaries = []

    #with tf.variable_scope('activations'):
    #
    #    histogram_summaries.append(tf.summary.histogram('log_probs', train_context['log_probs']))
    #
    #    def scrub_name(name, remove='/act_'):
    #        return name.replace(remove, '/')
    #
    #    histogram_summaries.extend(tf.summary.histogram(scrub_name(var.name), var)
    #                               for var in tf.get_collection(tf.GraphKeys.ACTIVATIONS))
    #
    #with tf.variable_scope('weights'):
    #    histogram_summaries.extend(tf.summary.histogram(var.name, var)
    #                               for var in tf.trainable_variables())

    with tf.variable_scope('losses'):
        loss_summaries = [tf.summary.scalar('train_loss', train_context['loss']),
                          tf.summary.scalar('val_loss', val_context['loss'])]

    loss_summary = tf.summary.merge([*loss_summaries, *histogram_summaries])

    if config.plot_step:

      with tf.variable_scope('samples'):
        summaries = []
        inputs = tf.placeholder(tf.float32, shape=None)
        outputs = tf.placeholder(tf.uint8, shape=None)
        num_samples = train_samples['triplets'].shape[0]

        summaries.append(tf.summary.image('outputs', outputs, max_outputs=num_samples))

        if 'image' in train_context:
            summaries.append(tf.summary.image('inputs', inputs, max_outputs=num_samples))
            input_images = train_context['image'][:train_samples['triplets'].shape[0]]
        else:
            input_images = tf.constant(0)

        plot_summary = tf.summary.merge(summaries)
        del summaries

        # Used to plot the ground truth
        true_triplets = train_context['triplets'][:train_samples['triplets'].shape[0]]


    saver = tf.train.Saver(var_list=model.variables,
                           keep_checkpoint_every_n_hours=6)


    print()
    print()


    ########################
    # Training the network #
    ########################


    with tf.Session(config=config.tf_config) as sess:

        sess.run(tf.global_variables_initializer())

        if config.model_dir.exists():
            print(f'Loading saved models from {config.model_dir}')
            saver.restore(sess, str(config.model_dir / 'model'))
        else:
            print(f'Starting from scratch in {config.model_dir}')
            config.model_dir.mkdir(parents=True, exist_ok=True)

        summary_writer = tf.summary.FileWriter(str(config.log_dir),
                                               session=sess)

        min_val_loss = val_loss_np = np.inf
        t0 = datetime.datetime.now()

        steps_it = range(config.training_steps)
        if not config.debug:
            steps_it = tqdm(steps_it)

        for n in steps_it:
            t1 = datetime.datetime.now()
            prefix = f'[{t1:%Y-%m-%d %H:%M:%S} step {n}]'
            spaces = ' ' * len(prefix)

            exec_file = config.out_dir / 'exec.py'
            if exec_file.exists():
                with tqdm.external_write_mode():
                    print(prefix, f'Executing {exec_file}')
                    try:
                        co = compile(exec_file.read_text(), str(exec_file), 'exec')
                    except Exception as e:
                        print(spaces, f'Compilation failed: {e}')
                    try:
                        exec(co)
                    except Exception as e:
                        print(spaces, f'Execution failed: {e}')

            if config.debug:
                print(prefix, 'Begin step')

            if n % config.loss_step == 0:
                summary_str, val_loss_np = sess.run((loss_summary, val_context['loss']))
                summary_writer.add_summary(summary_str, global_step=n)
                if not np.isfinite(val_loss_np):
                    log.error('non-finite validation loss %r', val_loss_np)
                    raise ValueError(f'validation loss is non-finite: {val_loss_np!r}')

            if config.save_step and (n % config.save_step) == 0 and n > 0:
                with tqdm.external_write_mode():
                    print(prefix, f'Saving model to {config.model_dir}')
                    print(spaces, f'Loop rate: {n/(t1-t0).total_seconds():.5g} iter/s')
                    saver.save(sess, str(config.model_dir / 'model'), global_step=n)
                    val_loss_np = sess.run(val_context['loss'])
                    print(spaces, f'Validation loss: {val_loss_np:.5g}')
                    if val_loss_np < min_val_loss:
                        min_val_loss = val_loss_np
                        print(spaces, 'Best model yet, putting aside')
                        saver.save(sess, str(config.model_dir / 'best'))

            if config.plot_step and n % config.plot_step == 0:

                with tqdm.external_write_mode():
                    print(prefix, 'Plotting samples')

                samples_np, true_trips_np, inputs_np = sess.run((train_samples, true_triplets, input_images))

                sampled_paths = []
                for i in range(len(samples_np['triplets'])):
                    triplets = samples_np['triplets'][i][:samples_np['num_triplets'][i]]
                    try:
                        path = triplets_to_path(triplets)
                    except ConversionError as err:
                        with tqdm.external_write_mode():
                            print(spaces, f'Failed to make path from sample {i}: {err}')
                        sampled_paths.append(None)
                    else:
                        sampled_paths.append(path)

                true_paths  = [triplets_to_path(true_trips_i_np) for true_trips_i_np in true_trips_np]

                outputs_np = []
                for sample, true in zip(sampled_paths, true_paths):
                    plt.clf()
                    plot_sample(sample, true)
                    plot_im = figure_to_ndarray()
                    outputs_np.append(plot_im)

                summary_str = sess.run(plot_summary, {inputs: inputs_np, outputs: outputs_np})
                summary_writer.add_summary(summary_str, global_step=n)
                summary_writer.flush()

            # Training step
            ops  = (optim_op, train_context['loss'])
            feed = {learning_rate: config.learning_rate}
            _, train_loss_np = sess.run(ops, feed)

            if hasattr(steps_it, 'set_postfix'):
                steps_it.set_postfix(tl=train_loss_np,
                                     vl=val_loss_np,
                                     vl_min=min_val_loss)


if __name__ == "__main__":
    main()
