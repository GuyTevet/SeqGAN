import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Gen_Data_loader_text8, Dis_dataloader, Dis_dataloader_text8
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import cPickle
import os
import collections
import json
from tqdm import tqdm

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 1 # 120 # supervise (maximum likelihood estimation) epochs for generator
DISC_PRE_EPOCH_NUM = 1 # 50 # supervise (maximum likelihood estimation) epochs for descriminator
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 1 #200 #num of adversarial epochs
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 100000 # 10000

#########################################################################################
#  Data configurations
#########################################################################################
use_real_world_data = True
real_data_file_path = './data/text8'


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    print 'Generating samples...'
    # Generate Samples
    generated_samples = []
    for _ in tqdm(range(int(generated_num / batch_size))):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def generate_real_data_samples(sess, trainable_model, batch_size, generated_num, output_file, inv_charmap):
    # Generate Samples
    print 'Generating real data samples...'
    generated_samples = []
    for _ in tqdm(range(int(generated_num / batch_size))):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ''.join([inv_charmap[x] for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in tqdm(xrange(data_loader.num_batch)):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def language_model_evaluation(sess, tested_model, data_loader):

    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        N = 1000 # hard coded for the meantime

        # estimate probability vectors for batch
        pred = np.zeros([BATCH_SIZE,SEQ_LENGTH,tested_model.num_emb],dtype=np.float32)
        for i in xrange(N):
            pred_one_hot, real_pred = tested_model.language_model_eval_step(sess, batch)
            pred += pred_one_hot
        pred = pred / (N * 1.)
        pred_flat = np.reshape(pred,[BATCH_SIZE * SEQ_LENGTH,tested_model.num_emb])
        batch_flat = np.reshape(batch,[BATCH_SIZE * SEQ_LENGTH])
        eps = 1e-10

        #calc bit per char
        BPC = np.average([-np.log2(pred_flat[i,batch_flat[i]] + eps) for i in xrange(pred_flat.shape[0])])
        print BPC

    return BPC

def split_text8(text8_orig_path):

    print('spliting text8 to train and test sets...')

    text8_train_path = text8_orig_path + '-train'
    text8_valid_path = text8_orig_path + '-valid'
    text8_test_path = text8_orig_path + '-test'

    # find each split size
    with open(text8_orig_path) as f:
        text8_size = len(f.read())
    assert text8_size == 100000000

    train_size = int(0.9 * text8_size)
    valid_size = int(0.05 * text8_size)
    test_size = int(0.05 * text8_size)

    with open(text8_orig_path,'r') as f_orig, \
            open(text8_train_path,'w') as f_train, \
            open(text8_valid_path, 'w') as f_valid, \
            open(text8_test_path,'w') as f_test:
        f_train.write(f_orig.read(train_size))
        f_valid.write(f_orig.read(valid_size))
        f_test.write(f_orig.read(test_size))

    return

def create_real_data_dict(data_file, dict_file):

    if not os.path.exists(dict_file): #create dict
        with open(data_file, 'r') as f:
            all_data = f.read()

        counts = collections.Counter(char for char in all_data)

        charmap = {}
        inv_charmap = []

        for char, count in counts.most_common(100):
            if char not in charmap:
                charmap[char] = len(inv_charmap)
                inv_charmap.append(char)

        assert len(charmap) == 27

        #save dict
        with open(dict_file,'w') as f:
            f.write(json.dumps(charmap))

    else: # load dict
        with open(dict_file, 'r') as f:
            charmap = json.loads(f.read())

        inv_charmap = [None] * len(charmap)
        for key in charmap.keys():
            inv_charmap[int(charmap[key])] = str(key)

    return charmap, inv_charmap
    

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    if use_real_world_data:
        vocab_size = 27
        # split to train-eval-test
        real_data_train_file = real_data_file_path + '-train'
        real_data_eval_file = real_data_file_path + '-eval'
        real_data_test_file = real_data_file_path + '-test'
        real_data_dict_file = real_data_file_path + '-dict.json'
        if not os.path.exists(real_data_train_file):
            split_text8(real_data_file_path)
        charmap, inv_charmap = create_real_data_dict(real_data_train_file,real_data_dict_file)
        gen_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap)
        dis_data_loader = Dis_dataloader_text8(BATCH_SIZE,charmap,inv_charmap)
    else:
        gen_data_loader = Gen_Data_loader(BATCH_SIZE)
        likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
        vocab_size = 5000
        dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    if not use_real_world_data:
        target_params = cPickle.load(open('save/target_params.pkl'))
        target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if use_real_world_data:
        gen_data_loader.create_batches(real_data_train_file)
    else:
        # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
        generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
        gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        print "start epoch %0d" % epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 1 == 0:
            if use_real_world_data:
                generate_real_data_samples(sess, generator, BATCH_SIZE, generated_num, eval_file + "_epoch_%0d.txt"%epoch ,inv_charmap)
                test_loss = 0 # FIXME - TEMP
            else:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)

            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)

    print 'Start pre-training discriminator...'
    # Train 3 epoch on the generated data and do this for 50 times
    for epoch in range(DISC_PRE_EPOCH_NUM):
        print "start epoch %0d"%epoch
        if use_real_world_data:
            generate_real_data_samples(sess, generator, BATCH_SIZE, generated_num , negative_file,inv_charmap)
            dis_data_loader.load_train_data(real_data_train_file, negative_file)
        else:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in tqdm(xrange(dis_data_loader.num_batch)):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            if not use_real_world_data:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                print 'total_batch: ', total_batch, 'test_loss: ', test_loss
                log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):

            if use_real_world_data:
                generate_real_data_samples(sess, generator, BATCH_SIZE, generated_num, negative_file, inv_charmap)
                dis_data_loader.load_train_data(real_data_train_file, negative_file)
            else:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in tqdm(xrange(dis_data_loader.num_batch)):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

    print '#########################################################################'
    print 'Start Language Model Evaluation...'
    test_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap)
    test_data_loader.create_batches(real_data_test_file)
    language_model_evaluation(sess,generator, test_data_loader)

    log.close()


if __name__ == '__main__':
    main()
