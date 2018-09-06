# -*- coding: utf-8 -*-
import re
import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from torch import nn
from gensim.models import KeyedVectors



def save_embed(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)
    print('Embedding saved')

def load_embed(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_en_embedding(word_dict, embedding_path, embedding_dim=300):
    """

    :param word_dict: vocabulary words' list
    :param embedding_path: pre-trained embedding path
    :param embedding_dim: embedding dimensions
    :return:
    """
    # find existing word embeddings
    word_vec = {}
    with open(embedding_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))

    print('Found {0}/{1} words with embedding vectors'.format(
        len(word_vec), len(word_dict)))
    missing_word_num = len(word_dict) - len(word_vec)
    missing_ratio = round(float(missing_word_num) / len(word_dict), 4) * 100
    print('Missing Ratio: {}%'.format(missing_ratio))

    # handling unknown embeddings
    for word in word_dict:
        if word not in word_vec:
            # If word not in word_vec, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            word_vec[word] = new_embedding
    print("Filled missing words' embeddings.")
    print("Embedding Matrix Size: ", len(word_vec))

    return word_vec


def get_cn_embeding(word_dict, full_embedding, embedding_dim=300):
    word_vec = {}
    count = 0

    for word in word_dict:
        if word in full_embedding.vocab:
            word_vec[word] = full_embedding[word]
            count += 1
        else:
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            word_vec[word] = new_embedding

    print('Found {0}/{1} words with full embedding vectors'.format(
        count, len(word_dict)))

    missing_word_num = len(word_dict) - count
    missing_ratio = round(float(missing_word_num) / len(word_dict), 4) * 100
    print('Missing Ratio: {}%'.format(missing_ratio))
    print("Filled missing words' embeddings.")
    print("Embedding Matrix Size: ", len(word_vec))

    return word_vec


def get_embedding(config, train_ds):
    src_lang = config['src_lang']

    """
    English Embedding
    """
    cur_en_embed_path = config['embedding']['cur_en_embedding_path']
    full_en_embed_path = config['embedding']['en_embed_path']

    if os.path.exists(cur_en_embed_path) and not config['make_dict']:
        en_embed = load_embed(cur_en_embed_path)
        print('Loaded existing english embedding, containing {} words.'.format(len(en_embed)))
    else:
        print('Making embedding...')
        en_embed = get_embedding(train_ds.tgt_vocab._id2word, full_en_embed_path)
        save_embed(en_embed, cur_en_embed_path)
        print('Saved generated embedding.')

    """
    Chinese Embedding
    """
    cur_cn_embed_path = config['embedding']['cur_cn_embedding_path']
    full_cn_embed_path = config['embedding']['cn_embed_path']

    if os.path.exists(cur_cn_embed_path) and not config['make_dict']:
        cn_embed = load_embed(cur_cn_embed_path)
        print('Loaded existing chinese embedding, containing {} words.'.format(len(cn_embed)))
    else:
        print('loading full w2v embeddings...')
        word_vectors = KeyedVectors.load_word2vec_format('data/sgns.merge.bigram.bz2')
        print('start extracting...')
        src_embed = get_cn_embeding(train_ds.src_vocab._id2word, word_vectors)
        save_embed(src_embed, 'data/cn_embed.pkl')

    if src_lang == 'chinese':
        src_embed = cn_embed
        tgt_embed = en_embed
    else:
        src_embed = en_embed
        tgt_embed = cn_embed

    src_vocab_size = len(src_embed)
    tgt_vocab_size = len(tgt_embed)

    # initialize nn embedding
    src_embedding = nn.Embedding(src_vocab_size, config['model']['embed_size'])
    tgt_embedding = nn.Embedding(tgt_vocab_size, config['model']['embed_size'])

    embed_list = []
    for word in train_ds.src_vocab._id2word:
        embed_list.append(src_embed[word])
    weight_matrix = np.array(embed_list)
    # pass weights to nn embedding
    src_embedding.weight = nn.Parameter(torch.from_numpy(weight_matrix).type(torch.FloatTensor), requires_grad=False)

    embed_list = []
    for word in train_ds.tgt_vocab._id2word:
        embed_list.append(tgt_embed[word])
    weight_matrix = np.array(embed_list)
    # pass weights to nn embedding
    tgt_embedding.weight = nn.Parameter(torch.from_numpy(weight_matrix).type(torch.FloatTensor), requires_grad=False)

    return src_embedding, src_vocab_size, tgt_embedding, tgt_vocab_size
