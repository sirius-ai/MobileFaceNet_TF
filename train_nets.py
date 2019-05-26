# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

from losses.face_losses import insightface_loss, cosineface_loss, combine_loss
from utils.data_process import parse_function, load_data
from nets.MobileFaceNet import inference
# from losses.face_losses import cos_loss
from verification import evaluate
from scipy.optimize import brentq
from utils.common import train
from scipy import interpolate
from datetime import datetime
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os

slim = tf.contrib.slim

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=12, help='epoch to train the network')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--class_number', type=int, required=True,
                        help='class number depend on your training datasets, MS1M-V1: 85164, MS1M-V2: 85742')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.', default=[4, 7, 9, 11])
    parser.add_argument('--train_batch_size', default=90, help='batch size to train network')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=100)
    parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    # parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--tfrecords_file_path', default='./datasets/faces_ms1m_112x112/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', default='./output/ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep ckpt files')
    #parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='', help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=2e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--loss_type', default='insightface', help='loss type, choice type are insightface/cosine/combine')
    parser.add_argument('--margin_s', type=float,
                        help='insightface_loss/cosineface_losses/combine_loss loss scale.', default=64.)
    parser.add_argument('--margin_m', type=float,
                        help='insightface_loss/cosineface_losses/combine_loss loss margin.', default=0.5)
    parser.add_argument('--margin_a', type=float,
                        help='combine_loss loss margin a.', default=1.0)
    parser.add_argument('--margin_b', type=float,
                        help='combine_loss loss margin b.', default=0.2)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args = get_parser()

        # create log dir
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
        # define placeholder
        inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='phase_train')

        # prepare train dataset
        # the image is substracted 127.5 and multiplied 1/128.
        # random flip left right
        tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
        dataset = tf.data.TFRecordDataset(tfrecords_f)
        dataset = dataset.map(parse_function)
        #dataset = dataset.shuffle(buffer_size=args.buffer_size)
        dataset = dataset.batch(args.train_batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # prepare validate datasets
        ver_list = []
        ver_name_list = []
        for db in args.eval_datasets:
            print('begin db %s convert.' % db)
            data_set = load_data(db, args.image_size, args)
            ver_list.append(data_set)
            ver_name_list.append(db)

        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        prelogits, net_points = inference(inputs, bottleneck_layer_size=args.embedding_size, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)

        # record the network architecture
        hd = open("./arch/txt/MobileFaceNet_Arch.txt", 'w')
        for key in net_points.keys():
            info = '{}:{}\n'.format(key, net_points[key].get_shape().as_list())
            hd.write(info)
        hd.close()

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-5
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        # inference_loss, logit = cos_loss(prelogits, labels, args.class_number)
        w_init_method = slim.initializers.xavier_initializer()
        if args.loss_type == 'insightface':
            inference_loss, logit = insightface_loss(embeddings, labels, args.class_number, w_init_method)
        elif args.loss_type == 'cosine':
            inference_loss, logit = cosineface_loss(embeddings, labels, args.class_number, w_init_method)
        elif args.loss_type == 'combine':
            inference_loss, logit = combine_loss(embeddings, labels, args.train_batch_size, args.class_number, w_init_method)
        else:
            assert 0, 'loss type error, choice item just one of [insightface, cosine, combine], please check!'
        tf.add_to_collection('losses', inference_loss)

        # total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')

        # define the learning rate schedule
        learning_rate = tf.train.piecewise_constant(epoch, boundaries=args.lr_schedule, values=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                                         name='lr_schedule')
        
        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # calculate accuracy
        pred = tf.nn.softmax(logit)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(args.summary_path, sess.graph)
        summaries = []
        # add train info to tensorboard summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('leraning_rate', learning_rate))
        summary_op = tf.summary.merge(summaries)

        # train op
        train_op = train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay,
                         tf.global_variables(), summaries, args.log_histograms)
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')

        # record trainable variable
        hd = open("./arch/txt/trainable_var.txt", "w")
        for var in tf.trainable_variables():
            hd.write(str(var))
            hd.write('\n')
        hd.close()

        # saver to load pretrained model or save model
        # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # output file path
        if not os.path.exists(args.log_file_path):
            os.makedirs(args.log_file_path)
        if not os.path.exists(args.ckpt_best_path):
            os.makedirs(args.ckpt_best_path)

        count = 0
        total_accuracy = {}
        for i in range(args.max_epoch):
            sess.run(iterator.initializer)
            _ = sess.run(inc_epoch_op)
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)

                    feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                    start = time.time()
                    _, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, regularization_losses, inc_global_step_op, Accuracy_Op],
                             feed_dict=feed_dict)
                    end = time.time()
                    pre_sec = args.train_batch_size/(end - start)

                    count += 1
                    # print training information
                    if count > 0 and count % args.show_info_interval == 0:
                        print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, reg_loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                              (i, count, total_loss_val, inference_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))

                    # save summary
                    if count > 0 and count % args.summary_interval == 0:
                        feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # save ckpt files
                    if count > 0 and count % args.ckpt_interval == 0:
                        filename = 'MobileFaceNet_iter_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)

                    # validate
                    if count > 0 and count % args.validate_interval == 0:
                        print('\nIteration', count, 'testing...')
                        for db_index in range(len(ver_list)):
                            start_time = time.time()
                            data_sets, issame_list = ver_list[db_index]
                            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
                            nrof_batches = data_sets.shape[0] // args.test_batch_size
                            for index in range(nrof_batches): # actual is same multiply 2, test data total
                                start_index = index * args.test_batch_size
                                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                                feed_dict = {inputs: data_sets[start_index:end_index, ...], phase_train_placeholder: False}
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
                            print('Equal Error Rate (EER): %1.3f\n' % eer)

                            with open(os.path.join(log_dir, '{}_result.txt'.format(ver_name_list[db_index])), 'at') as f:
                                f.write('%d\t%.5f\t%.5f\n' % (count, np.mean(accuracy), val))

                            if ver_name_list == 'lfw' and np.mean(accuracy) > 0.992:
                                print('best accuracy is %.5f' % np.mean(accuracy))
                                filename = 'MobileFaceNet_iter_best_{:d}'.format(count) + '.ckpt'
                                filename = os.path.join(args.ckpt_best_path, filename)
                                saver.save(sess, filename)

                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break
