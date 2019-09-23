#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: trainer.py
@time: 2019/5/23 16:45
"""

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from model import VAE,Discriminator
from timeit import default_timer as timer
import matplotlib
from torch.autograd import Variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import copy
from evaluator import Evaluator
from word_translation import get_word_translation_accuracy
from dico_builder import build_dictionary

class BiAVAE(object):
    def __init__(self, params):

        self.params = params
        self.tune_dir = "{}/{}-{}/{}".format(params.exp_id,params.src_lang,params.tgt_lang,params.norm_embeddings)
        self.tune_best_dir = "{}/best".format(self.tune_dir)
        
        if self.params.eval_file == 'wiki':
            self.eval_file = '/data/dictionaries/{}-{}.5000-6500.txt'.format(self.params.src_lang,self.params.tgt_lang)
        elif self.params.eval_file == 'wacky':
            self.eval_file = '/data/dictionaries/{}-{}.test.txt'.format(self.params.src_lang,self.params.tgt_lang)

        self.X_AE = VAE(params)
        self.Y_AE = VAE(params)
        self.D = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size,
                            output_size=params.d_output_size)

        self.nets = [self.X_AE, self.Y_AE, self.D]
        self.loss_fn = torch.nn.BCELoss()
        self.loss_fn2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def weights_init(self, m):  # 正交初始化
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0.01)

    def weights_init2(self, m):  # xavier_normal 初始化
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0.01)

    def weights_init3(self, m):  # 单位阵初始化
        if isinstance(m, torch.nn.Linear):
            m.weight.data.copy_(torch.diag(torch.ones(self.params.g_input_size)))

    def freeze(self, m):
        for p in m.parameters():
            p.requires_grad = False

    def defreeze(self, m):
        for p in m.parameters():
            p.requires_grad = True


    def init_state(self,state=1,seed=-1):
        if torch.cuda.is_available():
            # Move the network and the optimizer to the GPU
            for net in self.nets:
                net.cuda()
            self.loss_fn = self.loss_fn.cuda()
            self.loss_fn2 = self.loss_fn2.cuda()

        if state == 1:
            print('Init the model...')
            self.X_AE.apply(self.weights_init)  # 可更改G初始化方式
            self.Y_AE.apply(self.weights_init3)  # 可更改G初始化方式
            self.D.apply(self.weights_init2)
            self.Y_AE.apply(self.freeze)
            # self.X_AE.apply(self.freeze)


        elif state == 2:
            self.X_AE.load_state_dict(torch.load('{}/seed_{}_dico_{}_stage_1_best_X.t7'.format(self.tune_best_dir,seed,self.params.dico_build)))
            self.Y_AE.load_state_dict(torch.load('{}/seed_{}_dico_{}_stage_1_best_Y.t7'.format(self.tune_best_dir,seed,self.params.dico_build)))
            self.Y_AE.apply(self.defreeze)
            self.X_AE.apply(self.freeze)

            #self.D.load_state_dict(torch.load('{}/seed_{}_dico_{}_stage_1_best_D.t7'.format(self.tune_best_dir,seed,self.params.dico_build)))
            self.D.apply(self.weights_init2)

        elif state ==3:
            print('Init3 the model...')
            self.X_AE.apply(self.weights_init)  # 可更改G初始化方式
            self.Y_AE.apply(self.weights_init)  # 可更改G初始化方式
            self.D.apply(self.weights_init2)

        else:
            print('Invalid state!')

    def train(self, src_dico, tgt_dico, src_emb, tgt_emb, seed, stage):
        params = self.params
        # Load data
        if not os.path.exists(params.data_dir):
            print("Data path doesn't exists: %s" % params.data_dir)
        if not os.path.exists(self.tune_dir):
            os.makedirs(self.tune_dir)
        if not os.path.exists(self.tune_best_dir):
            os.makedirs(self.tune_best_dir)


        src_word2id = src_dico[1]
        tgt_word2id = tgt_dico[1]

        en = src_emb
        it = tgt_emb

        params = _get_eval_params(params)
        self.params = params
        eval = Evaluator(params, en,it, torch.cuda.is_available())


        AE_optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(self.X_AE.parameters()) + list(self.Y_AE.parameters())), lr=params.g_learning_rate)
        # AE_optimizer = optim.SGD(G_params, lr=0.1, momentum=0.9)
        # AE_optimizer = optim.Adam(G_params, lr=params.g_learning_rate, betas=(0.9, 0.9))
        # AE_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, list(self.X_AE.parameters()) + list(self.Y_AE.parameters())),lr=params.g_learning_rate,alpha=0.9)
        D_optimizer = optim.SGD(list(self.D.parameters()), lr=params.d_learning_rate)
        # D_optimizer = optim.Adam(D_params, lr=params.d_learning_rate, betas=(0.5, 0.9))
        # D_optimizer = optim.RMSprop(list(self.D_X.parameters()) + list(self.D_Y.parameters()), lr=params.d_learning_rate , alpha=0.9)

        # true_dict = get_true_dict(params.data_dir)
        D_acc_epochs = []
        d_loss_epochs = []
        G_AB_loss_epochs = []
        G_BA_loss_epochs = []
        G_AB_recon_epochs = []
        G_BA_recon_epochs = []
        g_loss_epochs = []
        acc_epochs = []

        csls_epochs = []
        best_valid_metric = -100

        # logs for plotting later
        log_file = open("log_src_tgt.txt", "w")  # Being overwritten in every loop, not really required
        log_file.write("epoch, disA_loss, disB_loss , disA_acc, disB_acc, g_AB_loss, g_BA_loss, g_AB_recon, g_BA_recon, CSLS, trans_Acc\n")

        if stage == 1:
            self.params.num_epochs = 50
        if stage == 2:
            self.params.num_epochs = 10

        try:
            for epoch in range(self.params.num_epochs):

                G_AB_recon = []
                G_BA_recon = []
                G_X_loss = []
                G_Y_loss = []
                d_losses = []
                g_losses = []
                hit_A = 0
                total = 0
                start_time = timer()
                # lowest_loss = 1e5
                label_D = to_variable(torch.FloatTensor(2 * params.mini_batch_size).zero_())
                label_D[:params.mini_batch_size] = 1 - params.smoothing
                label_D[params.mini_batch_size:] = params.smoothing

                label_G = to_variable(torch.FloatTensor(params.mini_batch_size).zero_())
                label_G = label_G + 1 - params.smoothing
                label_G2 = to_variable(torch.FloatTensor(params.mini_batch_size).zero_())+params.smoothing

                for mini_batch in range(0, params.iters_in_epoch // params.mini_batch_size):
                    for d_index in range(params.d_steps):
                        D_optimizer.zero_grad()  # Reset the gradients
                        self.D.train()

                        view_X, view_Y = self.get_batch_data_fast_new(en, it)

                        # Discriminator X
                        _,Y_Z = self.Y_AE(view_Y)
                        _,X_Z = self.X_AE(view_X)
                        Y_Z = Y_Z.detach()
                        X_Z = X_Z.detach()
                        input = torch.cat([Y_Z, X_Z], 0)

                        pred = self.D(input)
                        D_loss = self.loss_fn(pred, label_D)
                        D_loss.backward()  # compute/store gradients, but don't change params
                        d_losses.append(to_numpy(D_loss.data))

                        discriminator_decision_A = to_numpy(pred.data)
                        hit_A += np.sum(discriminator_decision_A[:params.mini_batch_size] >= 0.5)
                        hit_A += np.sum(discriminator_decision_A[params.mini_batch_size:] < 0.5)

                        D_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

                        # Clip weights
                        _clip(self.D, params.clip_value)

                        sys.stdout.write("[%d/%d] :: Discriminator Loss: %.3f \r" % (
                            mini_batch, params.iters_in_epoch // params.mini_batch_size,
                            np.asscalar(np.mean(d_losses))))
                        sys.stdout.flush()

                    total += 2* params.mini_batch_size * params.d_steps

                    for g_index in range(params.g_steps):
                        # 2. Train G on D's response (but DO NOT train D on these labels)
                        AE_optimizer.zero_grad()
                        self.D.eval()

                        view_X, view_Y = self.get_batch_data_fast_new(en, it)

                        # Generator X_AE
                        ## adversarial loss
                        X_recon, X_Z = self.X_AE(view_X)
                        Y_recon, Y_Z = self.Y_AE(view_Y)

                        # input = torch.cat([Y_Z, X_Z], 0)

                        predx = self.D(X_Z)
                        D_X_loss = self.loss_fn(predx, label_G)
                        predy = self.D(Y_Z)
                        D_Y_loss = self.loss_fn(predy, label_G2)

                        L_recon_X = 1.0 - torch.mean(self.loss_fn2(view_X, X_recon))
                        L_recon_Y = 1.0 - torch.mean(self.loss_fn2(view_Y, Y_recon))

                        G_loss = D_X_loss+D_Y_loss+L_recon_X+L_recon_Y

                        G_loss.backward()

                        g_losses.append(to_numpy(G_loss.data))
                        G_X_loss.append(to_numpy(D_X_loss.data+L_recon_X.data))
                        G_Y_loss.append(to_numpy(D_Y_loss.data+L_recon_Y.data))
                        G_AB_recon.append(to_numpy(L_recon_X.data))
                        G_BA_recon.append(to_numpy(L_recon_Y.data))

                        AE_optimizer.step()  # Only optimizes G's parameters

                        sys.stdout.write(
                            "[%d/%d] ::                                     Generator Loss: %.3f Generator Y recon: %.3f\r" % (
                                mini_batch, params.iters_in_epoch // params.mini_batch_size,
                                np.asscalar(np.mean(g_losses)),np.asscalar(np.mean(G_BA_recon))))
                        sys.stdout.flush()

                '''for each epoch'''

                D_acc_epochs.append(hit_A / total)
                G_AB_recon_epochs.append(np.asscalar(np.mean(G_AB_recon)))
                G_BA_recon_epochs.append(np.asscalar(np.mean(G_BA_recon)))
                d_loss_epochs.append(np.asscalar(np.mean(d_losses)))
                g_loss_epochs.append(np.asscalar(np.mean(g_losses)))

                print(
                    "Epoch {} : Discriminator Loss: {:.3f}, Discriminator Accuracy: {:.3f}, Generator Loss: {:.3f}, Time elapsed {:.2f} mins".
                        format(epoch, np.asscalar(np.mean(d_losses)), hit_A / total,
                               np.asscalar(np.mean(g_losses)),
                               (timer() - start_time) / 60))

                if (epoch + 1) % params.print_every == 0:
                    # No need for discriminator weights

                    _,X_Z = self.X_AE(Variable(en))
                    _,Y_Z = self.Y_AE(Variable(it))
                    X_Z = X_Z.data
                    Y_Z = Y_Z.data

                    mstart_time = timer()
                    for method in [params.eval_method]:
                        results = get_word_translation_accuracy(
                            params.src_lang, src_word2id, X_Z,
                            params.tgt_lang, tgt_word2id, Y_Z,
                            method=method,
                            dico_eval='default'
                        )
                        acc1 = results[0][1]

                    print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
                    print('Method:{} score:{:.4f}'.format(method, acc1))

                    csls = eval.dist_mean_cosine(X_Z, Y_Z)

                    if csls > best_valid_metric:
                        print("New csls value: {}".format(csls))
                        best_valid_metric = csls
                        fp = open(self.tune_best_dir + "/seed_{}_dico_{}_stage_{}_epoch_{}_acc_{:.3f}.tmp".format(seed,params.dico_build,stage,epoch, acc1), 'w')
                        fp.close()
                        torch.save(self.X_AE.state_dict(),self.tune_best_dir+'/seed_{}_dico_{}_stage_{}_best_X.t7'.format(seed,params.dico_build,stage))
                        torch.save(self.Y_AE.state_dict(),self.tune_best_dir+'/seed_{}_dico_{}_stage_{}_best_Y.t7'.format(seed,params.dico_build,stage))
                        torch.save(self.D.state_dict(),self.tune_best_dir+'/seed_{}_dico_{}_stage_{}_best_D.t7'.format(seed,params.dico_build,stage))

                    # Saving generator weights
                    fp = open(self.tune_dir + "/seed_{}_stage_{}_epoch_{}_acc_{:.3f}.tmp".format(seed,stage,epoch,acc1), 'w')
                    fp.close()

                    acc_epochs.append(acc1)
                    csls_epochs.append(csls)

            csls_fb, epoch_fb = max([(score, index) for index, score in enumerate(csls_epochs)])
            fp = open(self.tune_best_dir + "/seed_{}_dico_{}_stage_{}_epoch_{}_Acc_{:.3f}_{:.3f}.cslsfb".format(seed, params.dico_build,stage, epoch_fb,acc_epochs[epoch_fb],csls_fb), 'w')
            fp.close()

            # Save the plot for discriminator accuracy and generator loss
            # fig = plt.figure()
            # plt.plot(range(0, len(D_A_acc_epochs)), D_A_acc_epochs, color='b', label='D_A')
            # plt.plot(range(0, len(D_B_acc_epochs)), D_B_acc_epochs, color='r', label='D_B')
            # plt.ylabel('D_accuracy')
            # plt.xlabel('epochs')
            # plt.legend()
            # fig.savefig(self.tune_dir + '/seed_{}_stage_{}_D_acc.png'.format(seed, stage))
            #
            # fig = plt.figure()
            # plt.plot(range(0, len(D_A_loss_epochs)), D_A_loss_epochs, color='b', label='D_A')
            # plt.plot(range(0, len(D_B_loss_epochs)), D_B_loss_epochs, color='r', label='D_B')
            # plt.ylabel('D_losses')
            # plt.xlabel('epochs')
            # plt.legend()
            # fig.savefig(self.tune_dir + '/seed_{}_stage_{}_D_loss.png'.format(seed, stage))
            #
            # fig = plt.figure()
            # plt.plot(range(0, len(G_AB_loss_epochs)), G_AB_loss_epochs, color='b', label='G_AB')
            # plt.plot(range(0, len(G_BA_loss_epochs)), G_BA_loss_epochs, color='r', label='G_BA')
            # plt.ylabel('G_losses')
            # plt.xlabel('epochs')
            # plt.legend()
            # fig.savefig(self.tune_dir + '/seed_{}_stage_{}_G_loss.png'.format(seed,stage))
            #
            # fig = plt.figure()
            # plt.plot(range(0, len(G_AB_recon_epochs)), G_AB_recon_epochs, color='b', label='G_AB')
            # plt.plot(range(0, len(G_BA_recon_epochs)), G_BA_recon_epochs, color='r', label='G_BA')
            # plt.ylabel('G_recon_loss')
            # plt.xlabel('epochs')
            # plt.legend()
            # fig.savefig(self.tune_dir + '/seed_{}_stage_{}_G_Recon.png'.format(seed,stage))

            # fig = plt.figure()
            # plt.plot(range(0, len(L_Z_loss_epoches)), L_Z_loss_epoches, color='b', label='L_Z')
            # plt.ylabel('L_Z_loss')
            # plt.xlabel('epochs')
            # plt.legend()
            # fig.savefig(tune_dir + '/seed_{}_stage_{}_L_Z.png'.format(seed,stage))

            fig = plt.figure()
            plt.plot(range(0, len(acc_epochs)), acc_epochs, color='b', label='trans_acc1')
            plt.ylabel('trans_acc')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_stage_{}_trans_acc.png'.format(seed, stage))

            fig = plt.figure()
            plt.plot(range(0, len(csls_epochs)), csls_epochs, color='b', label='csls')
            plt.ylabel('csls')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_stage_{}_csls.png'.format(seed, stage))

            fig = plt.figure()
            plt.plot(range(0, len(g_loss_epochs)), g_loss_epochs, color='b', label='G_loss')
            plt.ylabel('g_loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_g_stage_{}_loss.png'.format(seed, stage))

            fig = plt.figure()
            plt.plot(range(0, len(d_loss_epochs)), d_loss_epochs, color='b', label='csls')
            plt.ylabel('D_loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_stage_{}_d_loss.png'.format(seed, stage))
            plt.close('all')


        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            torch.save(self.X_AE.state_dict(), 'X_model_interrupt.t7')
            torch.save(self.Y_AE.state_dict(), 'Y_model_interrupt.t7')
            torch.save(self.D.state_dict(), 'd_model_interrupt.t7')
            log_file.close()
            exit()

        log_file.close()
        return

    def get_batch_data_fast_new(self, emb_en, emb_it):

        params = self.params
        random_en_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        random_it_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        en_batch = to_variable(emb_en)[random_en_indices.cuda()]
        it_batch = to_variable(emb_it)[random_it_indices.cuda()]

        #print(random_en_indices)
        #print(random_it_indices)

        return en_batch, it_batch

    def export_dict(self,src_dico,tgt_dico,emb_en,emb_it,seed):
        params = self.params
        # Export adversarial dictionaries
        optim_X_AE = VAE(params).cuda()
        optim_Y_AE = VAE(params).cuda()
        print('Loading pre-trained models...')
        optim_X_AE.load_state_dict(torch.load(self.tune_dir + '/best/seed_{}_best_X.t7'.format(seed)))
        optim_Y_AE.load_state_dict(torch.load(self.tune_dir + '/best/seed_{}_best_Y.t7'.format(seed)))
        X_Z = optim_X_AE.encode(Variable(emb_en)).data
        Y_Z = optim_Y_AE.encode(Variable(emb_it)).data

        mstart_time = timer()
        for method in [params.eval_method]:
            results = get_word_translation_accuracy(
                params.src_lang, src_dico[1], X_Z,
                params.tgt_lang, tgt_dico[1], emb_it,
                method=method,
                dico_eval='default'
            )
            acc1 = results[0][1]
        for method in [params.eval_method]:
            results = get_word_translation_accuracy(
                params.tgt_lang, tgt_dico[1], Y_Z,
                params.src_lang, src_dico[1], emb_en,
                method=method,
                dico_eval='default'
            )
            acc2 = results[0][1]
        # csls = 0
        print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
        print('Method:{} score:{:.4f}-{:.4f}'.format(method, acc1, acc2))

        print('Building dictionaries...')
        params.dico_build = "S2T&T2S"
        params.dico_method = "csls_knn_10"
        X_Z = X_Z / X_Z.norm(2, 1, keepdim=True).expand_as(X_Z)
        emb_it = emb_it / emb_it.norm(2, 1, keepdim=True).expand_as(emb_it)
        f_dico_induce = build_dictionary(X_Z, emb_it, params)
        f_dico_induce = f_dico_induce.cpu().numpy()
        Y_Z = Y_Z / Y_Z.norm(2, 1, keepdim=True).expand_as(Y_Z)
        emb_en = emb_en / emb_en.norm(2, 1, keepdim=True).expand_as(emb_en)
        b_dico_induce = build_dictionary(Y_Z, emb_en, params)
        b_dico_induce = b_dico_induce.cpu().numpy()

        f_dico_set = set([(a,b) for a,b in f_dico_induce])
        b_dico_set = set([(b,a) for a,b in b_dico_induce])

        intersect = list(f_dico_set & b_dico_set)
        union = list(f_dico_set | b_dico_set)

        with io.open(self.tune_dir + '/best/{}-{}.dict'.format(params.src_lang, params.tgt_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in f_dico_induce:
                f.write('{} {}\n'.format(src_dico[0][item[0]], tgt_dico[0][item[1]]))

        with io.open(self.tune_dir + '/best/{}-{}.dict'.format(params.tgt_lang, params.src_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in b_dico_induce:
                f.write('{} {}\n'.format(tgt_dico[0][item[0]], src_dico[0][item[1]]))

        with io.open(self.tune_dir + '/best/{}-{}.intersect'.format(params.src_lang, params.tgt_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in intersect:
                f.write('{} {}\n'.format(src_dico[0][item[0]], tgt_dico[0][item[1]]))

        with io.open(self.tune_dir + '/best/{}-{}.intersect'.format(params.tgt_lang, params.src_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in intersect:
                f.write('{} {}\n'.format(tgt_dico[0][item[1]],src_dico[0][item[0]]))

        with io.open(self.tune_dir + '/best/{}-{}.union'.format(params.src_lang, params.tgt_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in union:
                f.write('{} {}\n'.format(src_dico[0][item[0]], tgt_dico[0][item[1]]))

        with io.open(self.tune_dir + '/best/{}-{}.union'.format(params.tgt_lang, params.src_lang), 'w', encoding='utf-8',
                     newline='\n') as f:
            for item in union:
                f.write('{} {}\n'.format(tgt_dico[0][item[1]], src_dico[0][item[0]]))

def _init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)

def to_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)

def to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()

def _clip(d, clip):
    if clip > 0:
        for x in d.parameters():
            x.data.clamp_(-clip, clip)


def _get_eval_params(params):
    params = copy.deepcopy(params)
    params.ks = [1]
    params.methods = ['csls']
    params.models = ['adv']
    params.refine = ['without-ref']

    params.dico_method = "csls_knn_10"
    # params.dico_build = "S2T&T2S"
    # params.dico_method = 'nn'
    # params.dico_build = "S2T"

    params.eval_method = "nn"
    params.dico_max_rank = 10000
    params.dico_max_size = 10000
    params.dico_min_size = 0
    params.dico_threshold = 0
    params.cuda = True
    params.d_learning_rate = 0.1
    params.d_steps = 5
    params.g_learning_rate = 0.1
    params.iters_in_epoch = 100000
    params.d_hidden_size = 2500
    params.mini_batch_size = 32
    params.g_size = 300
    return params
