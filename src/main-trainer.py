#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: main-trainer.py
@time: 2018/10/9 14:34
"""


from loader import load_embeddings
from trainer import BiAVAE
import torch
import argparse
import copy
import os
import numpy as np
from model import VAE,Discriminator
from torch.autograd import Variable
import random
from timeit import default_timer as timer
from word_translation import get_word_translation_accuracy
from dico_builder import build_dictionary
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for Unsupervised Bilingual Lexicon Induction using GANs')
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data'))
    parser.add_argument("--exp_id", type=str, default='tune')
    parser.add_argument("--state", type=int, default=1)

    parser.add_argument("--src_lang", type=str, default='en')
    parser.add_argument("--src_emb", type=str, default='')
    parser.add_argument("--tgt_lang", type=str, default='es')
    parser.add_argument("--tgt_emb", type=str, default='')
    parser.add_argument("--eval_file",type=str, default='')

    parser.add_argument("--g_input_size", type=int, default=300)
    parser.add_argument("--g_size", type=int, default=300)
    parser.add_argument("--g_output_size", type=int, default=300)
    parser.add_argument("--d_input_size", type=int, default=300)
    parser.add_argument("--d_hidden_size", type=int, default=2500)
    parser.add_argument("--d_output_size", type=int, default=1)
    parser.add_argument("--mini_batch_size", type=int, default=32)

    parser.add_argument("--dis_hidden_dropout", type=float, default=0)
    parser.add_argument("--dis_input_dropout",  type=float, default=0.1)

    parser.add_argument("--d_learning_rate", type=float, default=0.1)
    parser.add_argument("--g_learning_rate", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--d_steps", type=int, default=5)
    parser.add_argument("--g_steps", type=int, default=1)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--recon_weight", type=float, default=1)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--num_random_seeds", type=int, default=10)

    parser.add_argument("--iters_in_epoch", type=int, default=75000)
    parser.add_argument("--most_frequent_sampling_size", type=int,
                        default=75000)
    parser.add_argument("--print_every", type=int, default=1)
    # parser.add_argument("--lr_decay", type=float, default=0.98)
    # parser.add_argument("--lr_shrink", dest="lr_shrink", type=float, default=lr_shrink)
    # parser.add_argument("--lr_min", dest="lr_min", type=float, default=lr_min)
    # parser.add_argument("--add_noise", type=int, default=0)
    # parser.add_argument("--center_embeddings", dest="center_embeddings", type=int, default=center_embeddings)

    # parser.add_argument("--noise_mean", dest="noise_mean", type=float, default=noise_mean)
    # parser.add_argument("--noise_var", dest="noise_var", type=float, default=noise_var)

    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--top_frequent_words", type=int, default=200000)
    parser.add_argument("--max_vocab", type=int, default=200000)
    parser.add_argument("--csls_k", type=int, default=10)

    parser.add_argument("--beta", type=float, default=0.001)

    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--eval_method", type=str, default="nn")
    parser.add_argument("--dico_method",type=str , default="csls_knn_10")
    parser.add_argument("--dico_build",type=str , default="S2T&T2S")
    parser.add_argument("--dico_max_rank",type=int ,default=10000)
    parser.add_argument("--dico_max_size",type=int , default=10000)
    parser.add_argument("--dico_min_size",type=int , default=0)
    parser.add_argument("--dico_threshold",type=float , default=0)

    parser.add_argument("--refine_top", type=int, default=15000)
    parser.add_argument("--cosine_top", type=int, default=10000)
    parser.add_argument("--mask_procrustes", type=int, default=0)
    parser.add_argument("--num_refine", type=int, default=0)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--norm_embeddings", type=str, default='')

    parser.add_argument("--src_pretrain", type=str, default='')
    parser.add_argument("--tgt_pretrain", type=str, default='')
    parser.add_argument("--dataset", type=str, default='')



    return parser.parse_args()

def _get_eval_params(params):
    params = copy.deepcopy(params)
    params.ks = [1, 5, 10]
    params.methods = ['nn', 'csls']
    params.models = ['procrustes', 'adv']
    params.refine = ['without-ref', 'with-ref']
    return params

def initialize_exp(seed):
    if seed >= 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

def center_embeddings(emb):
    print('Centering the embeddings')
    mean = emb.mean(0)
    emb = emb-mean
    return emb
def norm_embeddings(emb):
    print('Normalizing the embeddings')
    norms = np.linalg.norm(emb,axis=1,keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    return emb

def main():
    params = parse_arguments()

    src_dico, src_emb = load_embeddings(params, source=True, full_vocab=False)
    tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=False)

    if params.norm_embeddings:
        norms = params.norm_embeddings.strip().split('_')
        for item in norms:
            if item == 'unit':
                src_emb = norm_embeddings(src_emb)
                tgt_emb = norm_embeddings(tgt_emb)
            elif item == 'center':
                src_emb = center_embeddings(src_emb)
                tgt_emb = center_embeddings(tgt_emb)
            elif item == 'none':
                pass
            else:
                print('Invalid norm:{}'.format(item))

    src_emb = torch.from_numpy(src_emb).float()
    tgt_emb = torch.from_numpy(tgt_emb).float()

    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda_device)
        src_emb = src_emb.cuda()
        tgt_emb = tgt_emb.cuda()
    
    if params.dataset == 'wiki':
        params.eval_file = '/data/dictionaries/{}-{}.5000-6500.txt'.format(params.src_lang, params.tgt_lang)
    elif params.dataset == 'wacky':
        params.eval_file = '/data/dictionaries/{}-{}.test.txt'.format(params.src_lang, params.tgt_lang)
    else:
        params.eval_file = ''

    if params.mode == 0:  # train model
        seed_list = [864,394,776,911,430,41,265,988,523,497]
        #seed_list = [430]
        for item in seed_list:
            # init_seed = random.randint(0, 1000)
            init_seed = item
            # t = BiAAE(params)
            initialize_exp(init_seed)
            t = BiAVAE(params)
            t.init_state(state=params.state,seed=init_seed)
            t.train(src_dico,tgt_dico,src_emb,tgt_emb,init_seed,stage=params.state)
            t.init_state(state=2, seed=init_seed)
            t.train(src_dico, tgt_dico, src_emb, tgt_emb, init_seed, stage=2)
            # t.export_dict(src_dico,tgt_dico,src_emb,tgt_emb,seed)
    elif params.mode ==1:

        X_AE = VAE(params).cuda()
        Y_AE = VAE(params).cuda()

        X_AE.load_state_dict(torch.load(params.src_pretrain))
        Y_AE.load_state_dict(torch.load(params.tgt_pretrain))

        X_Z = X_AE(Variable(src_emb))[1].data
        Y_Z = Y_AE(Variable(tgt_emb))[1].data

        mstart_time = timer()
        for method in ['nn', 'csls_knn_10']:
            results = get_word_translation_accuracy(
                params.src_lang, src_dico[1], X_Z,
                params.tgt_lang, tgt_dico[1], Y_Z,
                method=method,
                dico_eval='default' if not params.eval_file else params.eval_file,
                device=params.cuda_device
            )
            acc = results[0][1]
            print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
            print('Method:{} score:{:.4f}'.format(method, acc))
        '''
        params.dico_build = "S2T&T2S"
        params.dico_method = "csls_knn_10"
        X_Z = X_Z / X_Z.norm(2, 1, keepdim=True).expand_as(X_Z)
        Y_Z = Y_Z / Y_Z.norm(2, 1, keepdim=True).expand_as(Y_Z)
        f_dico_induce = build_dictionary(X_Z, Y_Z, params)

        # induced_dict = [(a, b) for a, b in f_dico_induce]
        with io.open('dict/{}-{}.dict'.format(params.src_lang, params.tgt_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in f_dico_induce:
                f.write('{} {}\n'.format(src_dico[0][item[0]], tgt_dico[0][item[1]]))
        '''
    else:
        print("Invalid flag!")

if __name__ == '__main__':
    main()
