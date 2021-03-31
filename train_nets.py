# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

from losses.face_losses import insightface_loss, cosineface_loss, combine_loss
from utils.data_process import parse_function, load_data
from nets.MobileFaceNet import inference
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
    parser.add_argument('--max_epoch', default=12, type=int, help='epoch to train the network')
    parser.add_argument('--image_size', default=[112, 112], type=list, help='the image size')
    parser.add_argument('--class_number', type=int, required=True,
                        help='class number depend on your training datasets, MS1M-V1: 85164, MS1M-V2: 85742')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    # Difference from original paper implementation
    # "We set the weight decay parameter to be 4e-5, except the
    # weight decay parameter of the last layers after the global operator (GDConv or
    # GAPool) being 4e-4"
    parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 weight regularization.')
    #
    parser.add_argument('--lr_schedule', type=float, help='Number of epochs for learning rate piecewise.', default=[4, 9, 14, 19])
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size to train network')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=128)
    parser.add_argument('--eval_datasets', type=list, default=['lfw', 'agedb_30'], help='evaluation datasets')
    
    # ???????????????????????
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evaluate datasets base path')
    # ???????????????????????
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--tfrecords_file_path', default='./datasets/faces_ms1m_112x112/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', type=str, default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', type=str, default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', type=str, default='./output/ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--log_file_path', type=str, default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', type=int, default=50, help='tf.train.Saver max keep ckpt files')
    #parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', type=int, default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', type=int, default=2000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', type=int, default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', type=int, default=50, help='intervals to show info')
    # -----------------------------------
    parser.add_argument('--pretrained_model', type=str, default=None, help='Load a pretrained model before training starts.')
    # Difference from original paper:
    # SGD with momentum 0.9 ?
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    # ?
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    # ?
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    # ?
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=2e-5)
    # ?
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


def main():
    cur_time= datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    print(f'\n\n\n***TRAINING SESSION START AT {cur_time}***\n\n\n')
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args = get_parser()

        # define global params
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch_step= tf.Variable(name='epoch_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=0, trainable=False)
        
        # def placeholders
        print(f'***Input of size: {args.image_size}')
        print(f'***Perform evaluation after each {args.validate_interval} on datasets: {args.eval_datasets}')
        inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='phase_train')

        # prepare train dataset
        # the image is substracted 127.5 and multiplied 1/128.
        # random flip left right
        tfrecords_f = os.path.join(args.tfrecords_file_path, 'train.tfrecords')
        dataset = tf.data.TFRecordDataset(tfrecords_f)
        dataset = dataset.map(parse_function)
        # dataset = dataset.shuffle(buffer_size=args.buffer_size)
        dataset = dataset.batch(args.train_batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        prelogits, net_points = inference(inputs, bottleneck_layer_size=args.embedding_size, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)
        # record the network architecture
        hd = open("./arch/txt/MobileFaceNet_architecture.txt", 'w')
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
            print(f'INSIGHTFACE LOSS WITH s={args.margin_s}, m={args.margin_m}')
            inference_loss, logit = insightface_loss(embeddings, labels, args.class_number, w_init_method, s= args.margin_s, m=args.margin_m)
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

        # calculate accuracy op
        pred = tf.nn.softmax(logit)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(args.summary_path, sess.graph)
        summaries = []
        # add train info to tensorboard summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))
        summaries.append(tf.summary.scalar('training_acc', Accuracy_Op))
        summary_op = tf.summary.merge(summaries)

        # train op
        train_op = train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay,
                         tf.global_variables(), summaries, args.log_histograms)
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        inc_epoch_step_op= tf.assign_add(epoch_step, 1, name='increment_epoch_step')
        reset_epoch_step_op=tf.assign(epoch_step, 0, name='reset_epoch_step')
        inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')

        # record trainable variable
        hd = open("./arch/txt/trainable_var.txt", "w")
        for var in tf.trainable_variables():
            hd.write(str(var))
            hd.write('\n')
        hd.close()

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # RELOAD CHECKPOINT FOR PRETRAINED MODEL
        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('***Pre-trained model: %s' % pretrained_model)
        
        if pretrained_model is None:
            # saver to load pretrained model or save model
            saver = tf.train.Saver(tf.trainable_variables()+ [epoch, epoch_step, global_step], max_to_keep=args.saver_maxkeep)
        else:
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)
        # lask checkpoint path
        checkpoint_path = None
        if args.ckpt_path:
            ckpts=os.listdir(args.ckpt_path)
            if 'checkpoint' in ckpts:
              ckpts.remove('checkpoint')
            ckpts_prefix=[x.split('_')[0] for x in ckpts]
            ckpts_prefix.sort(key=lambda x: int(x) , reverse=True)
            
            # Get last checkpoint
            if len(ckpts_prefix)>0:
                last_ckpt= f"{ckpts_prefix[0]}_MobileFaceNet.ckpt"
                checkpoint_path = os.path.expanduser(os.path.join(args.ckpt_path, last_ckpt))
                print('***Last checkpoint: %s' % checkpoint_path)

        # load checkpoint model
        if checkpoint_path is not None:
            print('***Restoring checkpoint: %s' % checkpoint_path)
            saver.restore(sess, checkpoint_path)
        # load pretrained model
        elif pretrained_model:
            print('***Restoring pretrained model: %s' % pretrained_model)
            # ckpt = tf.train.get_checkpoint_state(pretrained_model)
            # print(ckpt)
            saver.restore(sess, pretrained_model)
        else:
            print('***No checkpoint or pretrained model found.')
            print('***Training from scratch')

        
        # output file path
        if not os.path.exists(args.log_file_path):
            os.makedirs(args.log_file_path)
        if not os.path.exists(args.ckpt_best_path):
            os.makedirs(args.ckpt_best_path)
        
        # prepare validate datasets
        ver_list = []
        ver_name_list = []
        print('***LOADING VALIDATION DATABASES..')
        for db in args.eval_datasets:
            print('\t- Loading database: %s' % db)
            data_set = load_data(db, args.image_size, args)
            ver_list.append(data_set)
            ver_name_list.append(db)

        cur_epoch, cur_global_step, cur_epoch_step = sess.run([epoch, global_step, epoch_step])
        print('****************************************')
        print(f'Continuous training on EPOCH={cur_epoch}, GLOBAL_STEP={cur_global_step}, EPOCH_STEP={cur_epoch_step}')
        print('****************************************')

        total_losses_per_summary = []
        inference_losses_per_summary = []
        train_acc_per_summary = []
        avg_total_loss_per_summary = 0
        avg_inference_loss_per_summary = 0
        avg_train_acc_per_summary = 0
        for i in range(cur_epoch, args.max_epoch+1):
            sess.run(iterator.initializer)
            # Trained steps are ignored
            print(f'Skipping {cur_epoch_step} trained step..')
            start = time.time()
            for _j in range(cur_epoch_step):
                images_train, labels_train = sess.run(next_element)
                if _j % 1000 == 0:
                    end = time.time()
                    iter_time = end - start
                    start = time.time()
                    print(f'{_j}, time: {iter_time} seconds')
            print('***Traing started***')
            while True:
                try:
                    start = time.time()
                    images_train, labels_train = sess.run(next_element)
                    feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                    _, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, regularization_losses, inc_epoch_step_op, Accuracy_Op],
                             feed_dict=feed_dict)
                    end = time.time()
                    pre_sec = args.train_batch_size/(end - start)

                    cur_global_step += 1
                    cur_epoch_step +=1

                    total_losses_per_summary.append(total_loss_val)
                    inference_losses_per_summary.append(inference_loss_val)
                    train_acc_per_summary.append(acc_val)

                    # print training information
                    if cur_global_step > 0 and cur_global_step % args.show_info_interval == 0:
                        print('epoch %d, total_step %d, epoch_step %d, total loss %.2f , inference loss %.2f, reg_loss %.2f, training accuracy %.6f, rate %.3f samples/sec' %
                              (i, cur_global_step, cur_epoch_step, total_loss_val, inference_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))

                    # save summary
                    if cur_global_step > 0 and cur_global_step % args.summary_interval == 0:
                        feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, cur_global_step)

                        avg_total_loss_per_summary = sum(total_losses_per_summary)/len(total_losses_per_summary)
                        total_losses_per_summary = []
                        avg_inference_loss_per_summary = sum(inference_losses_per_summary) / len(inference_losses_per_summary)
                        inference_losses_per_summary = []
                        avg_train_acc_per_summary = sum(train_acc_per_summary) / len(train_acc_per_summary)
                        train_acc_per_summary = []
                        # Create a new Summary object with your measure
                        summary2 = tf.Summary()
                        summary2.value.add(tag='avg_total_loss', simple_value=avg_total_loss_per_summary)
                        summary2.value.add(tag='avg_inference_loss', simple_value = avg_inference_loss_per_summary)
                        summary2.value.add(tag='avg_train_acc', simple_value = avg_train_acc_per_summary)
                   

                        # Add it to the Tensorboard summary writer
                        # Make sure to specify a step parameter to get nice graphs over time
                        summary.add_summary(summary2, cur_global_step)


                    # save ckpt files
                    if cur_global_step > 0 and cur_global_step % args.ckpt_interval == 0:
                        filename = '{:d}_MobileFaceNet'.format(cur_global_step) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)

                    # validate
                    if cur_global_step > 0 and cur_global_step % args.validate_interval == 0:
                        print('-------------------------------------------------')
                        print('\nIteration', cur_global_step, 'validating...')
                        for db_index in range(len(ver_list)):
                            start_time = time.time()
                            data_sets, issame_list = ver_list[db_index]
                            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
                            if data_sets.shape[0] % args.test_batch_size ==0:
                            	nrof_batches = data_sets.shape[0] // args.test_batch_size
                            else:
                            	nrof_batches = data_sets.shape[0] // args.test_batch_size +1
                            for index in range(nrof_batches): # actual is same multiply 2, test data total
                                start_index = index * args.test_batch_size
                                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                                feed_dict = {inputs: data_sets[start_index:end_index, ...], phase_train_placeholder: False}
                                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)
                            duration = time.time() - start_time

                            print("---Total time %.3fs to evaluate %d images of %s" % (duration, data_sets.shape[0], ver_name_list[db_index]))
                            print('\t- Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                            print('\t- Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                            print('\t- FPR and TPR: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

                            auc = metrics.auc(fpr, tpr)
                            print('\t- Area Under Curve (AUC): %1.3f' % auc)
                            # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                            # print('Equal Error Rate (EER): %1.3f\n' % eer)

                            with open(os.path.join(args.log_file_path, '{}_result.txt'.format(ver_name_list[db_index])), 'at') as f:
                                f.write('%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' % (cur_global_step, np.mean(accuracy), val, val_std, far, auc))

                            if ver_name_list[db_index] == 'lfw' and np.mean(accuracy) > 0.994:
                                print('High accuracy: %.5f' % np.mean(accuracy))
                                filename = 'MobileFaceNet_iter_best_{:d}'.format(cur_global_step) + '.ckpt'
                                filename = os.path.join(args.ckpt_best_path, filename)
                                saver.save(sess, filename)
                            print('---------------------------------------------------')

                except tf.errors.OutOfRangeError:
                    _, _ = sess.run([inc_epoch_op, reset_epoch_step_op])
                    # Save checkpoint
                    filename = '{:d}_MobileFaceNet'.format(cur_global_step) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)
                    cur_epoch_step=0
                    print("\n\n-------End of epoch %d\n\n" % i)
                    break


if __name__ == '__main__':
    main()
