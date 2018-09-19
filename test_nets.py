# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.data_process import load_data
from verification import evaluate
from scipy.optimize import brentq
from scipy import interpolate
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import re
import os


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # prepare validate datasets
            ver_list = []
            ver_name_list = []
            for db in args.eval_datasets:
                print('begin db %s convert.' % db)
                data_set = load_data(db, args.image_size, args)
                ver_list.append(data_set)
                ver_name_list.append(db)

            # Load the model
            load_model(args.model)

            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            embedding_size = embeddings.get_shape()[1]

            for db_index in range(len(ver_list)):
                # Run forward pass to calculate embeddings
                print('\nRunnning forward pass on {} images'.format(ver_name_list[db_index]))
                start_time = time.time()
                data_sets, issame_list = ver_list[db_index]
                nrof_batches = data_sets.shape[0] // args.test_batch_size
                emb_array = np.zeros((data_sets.shape[0], embedding_size))

                for index in range(nrof_batches):
                    start_index = index * args.test_batch_size
                    end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                    feed_dict = {inputs_placeholder: data_sets[start_index:end_index, ...]}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)
                duration = time.time() - start_time
                print("total time %.3fs to evaluate %d images of %s" % (duration, data_sets.shape[0], ver_name_list[db_index]))
                print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

                auc = metrics.auc(fpr, tpr)
                print('Area Under Curve (AUC): %1.3f' % auc)
                eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                print('Equal Error Rate (EER): %1.3f' % eer)

def parse_arguments(argv):
    '''test parameters'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='./output/ckpt')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=100)
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))