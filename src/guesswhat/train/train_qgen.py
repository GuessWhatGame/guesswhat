import argparse
import json
import os

import tensorflow as tf

from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM

import guesswhat.data_provider as provider

from tqdm import tqdm

if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Oracle network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-config", type=str, help="Configuration file")
    parser.add_argument("-seed", type=int, help='Seed', default=-1)
    parser.add_argument("-from_checkpoint", type=bool, required=False, help="Load a pre-trained model")
    parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")

    args = parser.parse_args()

    config, env, save_path, exp_identifier, logger = provider.load_data_from_args(args, load_picture=True)
    
    ###############################
    #  START TRAINING
    #############################

    logger.info('Building network..')
    network = QGenNetworkLSTM(config)


    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        if args.from_checkpoint:
            saver.restore(sess, args.from_checkpoint)
        else:
            sess.run(tf.global_variables_initializer())

        best_val_loss = 1e5
        for t in range(0, config['optimizer']['no_epoch']):

            logger.info('Epoch {}..'.format(t + 1))
            one_sample = {
                "picture_fc8": env.trainset.games[0].picture.fc8.reshape((1, 1000)),
            }
            # warning! the beam search stops after detecting the Stop token :)
            # we can either remove this constraint or use beam.word_id property (cf use_case of train_loop)
            beam_sequence, tmp = network.eval_one_beam_search(sess, one_sample, env.tokenizer, max_depth=50)
            logger.info(env.tokenizer.decode(tmp))

            iterator = provider.GameIterator(
                env.trainset,
                env.tokenizer,
                batch_size=config['optimizer']['batch_size'],
                shuffle=True,
                status=('success',))

            train_loss = 0.0

            for N, batch in enumerate(tqdm(iterator)):
                l, _ = sess.run([network.ml_loss, network.ml_optimize],
                                feed_dict={
                                    network.picture_fc8: batch['picture_fc8'],
                                    network.dialogues: batch['padded_tokens'],
                                    network.answer_mask: batch['answer_mask'],
                                    network.padding_mask: batch['padding_mask'],
                                    network.seq_length: batch['seq_length']
                                })
                train_loss += l
            train_loss /= (N+1)

            # Validation
            iterator = provider.GameIterator(
                env.validset,
                env.tokenizer,
                batch_size=config['optimizer']['batch_size'],
                shuffle=True,
                status=('success',))

            valid_loss = 0.0

            for N, batch in enumerate(tqdm(iterator)):
                l, = sess.run([network.ml_loss],
                              feed_dict={
                                    network.picture_fc8: batch['picture_fc8'],
                                    network.dialogues: batch['padded_tokens'],
                                    network.answer_mask: batch['answer_mask'],
                                    network.padding_mask: batch['padding_mask'],
                                    network.seq_length: batch['seq_length']
                                })
                valid_loss += l
            valid_loss /= (N+1)

            logger.info("Training loss:  {}".format(train_loss))
            logger.info("Validation loss:  {}".format(valid_loss))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                saver.save(sess, save_path.format('params.ckpt'))

        # Experiment done; write results to experiment database (jsonl file)
        with open(os.path.join(args.exp_dir, 'experiments.jsonl'), 'a') as f:
            exp = dict()
            exp['config'] = config
            exp['best_val_loss'] = best_val_loss
            exp['identifier'] = exp_identifier

            f.write(json.dumps(exp))
            f.write('\n')

        # Compute on Test
        saver.restore(sess, save_path.format('params.ckpt'))

        iterator = provider.GameIterator(
            env.trainset,
            env.tokenizer,
            batch_size=config['optimizer']['batch_size'],
            shuffle=True,
            status=('success',))

        test_loss = 0
        for N, batch in enumerate(tqdm(iterator)):
            l, = sess.run([network.ml_loss],
                          feed_dict={
                                    network.picture_fc8: batch['picture_fc8'],
                                    network.dialogues: batch['padded_tokens'],
                                    network.answer_mask: batch['answer_mask'],
                                    network.padding_mask: batch['padding_mask'],
                                    network.seq_length: batch['seq_length']
                                })
            test_loss += l
        test_loss /= (N+1)

        logger.info("Summary:")
        logger.info("validation loss: {}".format(best_val_loss))
        logger.info("Test loss: {}".format(test_loss))
