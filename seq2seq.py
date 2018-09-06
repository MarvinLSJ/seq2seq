# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
from datetime import datetime

from io import open
import unicodedata
import string
import re
import os
import random
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

from utils import load_embed, save_embed, get_embedding
from data import myDS
from models import LSTMEncoder, AttnLSTMDecoder

from data import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



FLAGS = None

def main(_):

    # Load the configuration file.
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    print('**********', config['experiment_name'],'**********')

    """ Data Preprocessing """

    # if config['data_preprocessing']:
    #     print 'Pre-processing Original Data ...'
    #     data_preprocessing()
    #     print 'Data Pre-processing Done!'

    """ Read Data """

    cn = pd.read_csv('data/cn_split.csv')
    en = pd.read_csv('data/en.csv')
    pair = pd.concat([cn, en], axis=1)

    # split dataset
    msk = np.random.rand(len(pair)) < 0.9
    train = pair[msk]
    valid = pair[~msk]

    src_lang = config['src_lang']
    tgt_lang = config['tgt_lang']

    # All senteneces (including train and valid)
    src_sents = pair[src_lang].tolist()
    tgt_sents = pair[tgt_lang].tolist()

    train_ds = myDS(train, src_lang, tgt_lang, src_sents, tgt_sents)
    valid_ds = myDS(valid, src_lang, tgt_lang, src_sents, tgt_sents)

    print('\nPreparing {} - {} NMT Model.\n'.format(src_lang, tgt_lang))
    print(
        'Preparing {} Training sentence pairs; {} Validation pairs.\n\nWith {} source language words; {} target language words.\n'.format(
            train_ds.__len__(), valid_ds.__len__(), len(train_ds.src_vocab._id2word), len(train_ds.tgt_vocab._id2word)))

    """ Get Embedding  """
    # embedding
    config['src_embedding_matrix'], config['src_vocab_size'], config['tgt_embedding_matrix'], config[
        'tgt_vocab_size'] = get_embedding(config, train_ds)

    all_costs = []

    # model
    enc = LSTMEncoder(config)
    dec = AttnLSTMDecoder(config)

    # data loader
    train_dataloader = DataLoader(dataset=train_ds, shuffle=True, batch_size=config['model']['batch_size'], drop_last=True)
    teacher_forcing_ratio = 1.0

    # loss
    criterion = nn.NLLLoss()

    # optimizer
    enc_optimizer = optim.SGD(enc.parameters(), lr=config['training']['learning_rate'])
    dec_optimizer = optim.SGD(dec.parameters(), lr=config['training']['learning_rate'])

    SOS_TOKEN = train_ds.tgt_vocab._word2id[train_ds.tgt_vocab.sos_token]
    EOS_TOKEN = train_ds.tgt_vocab._word2id[train_ds.tgt_vocab.eos_token]

    best_record = 100.0

    ckpt_path = os.path.join(config['ckpt_dir'], config['experiment_name']+'.pt')

    # load existing models
    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        enc.load_state_dict(ckpt['encoder'])
        dec.load_state_dict(ckpt['decoder'])
        enc_optimizer.load_state_dict(ckpt['encoder_optimizer'])
        dec_optimizer.load_state_dict(ckpt['decoder_optimizer'])
    else:
        epoch = 1
        print('Fresh start!\n')


    while epoch < config['training']['num_epochs']:

        # Train
        print('Start Epoch {} Training...'.format(epoch))

        train_loss = []

        for idx, data in enumerate(train_dataloader, 0):

            src = data[0]
            tgt = data[1]

            loss = 0

            # Encoder
            enc_outputs = torch.zeros(config['max_length'], enc.hidden_size, device=device)
            enc_h, enc_c = enc.initHiddenCell()
            for i in range(len(src)):
                enc_out, enc_h, enc_c = enc(src[i], enc_h, enc_c)
                if i >= config['max_length']:
                    break
                enc_outputs[i] = enc_out[0, 0]

            dec_in = torch.tensor(SOS_TOKEN, device=device).repeat(config['model']['batch_size'])
            dec_h = enc_h
            dec_c = enc_c

            # Decoder

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for j in range(len(tgt) - 1):
                    dec_out, dec_h, dec_att = dec(dec_in, dec_h, dec_c, enc_outputs)
                    loss += criterion(dec_out, tgt[j + 1])
                    dec_in = tgt[j + 1]  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for j in range(len(tgt) - 1):
                    dec_out, dec_h, dec_att = dec(dec_in, dec_h, dec_c, enc_outputs)

                    topv, topi = dec_out.topk(1)

                    dec_in = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(dec_out, tgt[j + 1])
            #             if dec_in == dec.embedding(torch.tensor(EOS_TOKEN, device=device)):
            #                 break

            train_loss.append(loss.data[0])

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            loss.backward()

            enc_optimizer.step()
            dec_optimizer.step()

            if (len(train_loss)) % 1000 == 0:
                print("{}/{} loss: {} ".format(idx + 1, len(train_ds.src), round(np.mean(train_loss), 4)))

        # Valid
        print('Epoch {} Validating...'.format(epoch))

        # loss
        valid_loss = []

        # dataloader
        valid_dataloader = DataLoader(dataset=valid_ds, shuffle=True,
                                      batch_size=config['model']['batch_size'], drop_last=True)

        for idx, data in enumerate(valid_dataloader, 0):
            src = data[0]
            tgt = data[1]

            # Encoder
            enc_outputs = torch.zeros(config['max_length'], enc.hidden_size, device=device)
            enc_h, enc_c = enc.initHiddenCell()
            for i in range(len(src)):
                enc_out, enc_h, enc_c = enc(src[i], enc_h, enc_c)
                enc_outputs[i] = enc_out[0, 0]

            dec_in = torch.tensor(SOS_TOKEN, device=device).repeat(64)
            dec_h = enc_h
            dec_c = enc_c

            for j in range(len(tgt) - 1):
                dec_out, dec_h, dec_att = dec(dec_in, dec_h, dec_c, enc_outputs)
                loss += criterion(dec_out, tgt[j + 1])
                dec_in = tgt[j + 1]  # Teacher forcing

            valid_loss.append(loss.data[0])

        print('Epoch {} Validation Loss: {}'.format(epoch, np.mean(valid_loss)))

        epoch += 1

        # Keep track of best record
        if np.mean(valid_loss) < best_record:
            best_record = np.mean(valid_loss)
            # save the best model
            state_dict = {
                'epoch': epoch,
                'encoder': enc.state_dict(),
                'decoder': dec.state_dict(),
                'encoder_optimizer': enc_optimizer.state_dict(),
                'decoder_optimizer': dec_optimizer.state_dict(),
            }
            torch.save(state_dict, ckpt_path)
            print('Model saved!\n')

    # """ Inference """
    #
    # if config['taks'] == 'inference':
    #     testDS = mytestDS(test_data, all_sents)
    #     # Do not shuffle here
    #     test_dataloader = DataLoader(dataset=testDS, num_workers=2, batch_size=1)
    #
    #     result = []
    #     for idx, data in enumerate(test_dataloader, 0):
    #
    #         # get data
    #         s1, s2 = data
    #
    #         # input
    #         output = siamese(s1,s2)
    #         output = output.squeeze(0)
    #
    #         # feed output into softmax to get prob prediction
    #         sm = nn.Softmax(dim=1)
    #         res = sm(output.data)[:,1]
    #         result += res.data.tolist()
    #
    #     result = pd.DataFrame(result)
    #     print 'Inference Done.'
    #     res_path = os.path.join(config['result']['filepath'], config['result']['filename'])
    #     result.to_csv(res_path, header=False, index=False)
    #     print 'Result has writtn to', res_path, ', Good Luck!'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file.')
    FLAGS, _ = parser.parse_known_args()
    main(_)