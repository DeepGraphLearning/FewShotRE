from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder
import models
from models.regrab import REGRAB
from models.d import Discriminator
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import random
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_wiki',
            help='val file')
    parser.add_argument('--test', default='test_wiki',
            help='test file')
    parser.add_argument('--adv', default=None,
            help='adv file')
    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=5, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='regrab',
            help='model name')
    parser.add_argument('--encoder', default='bert',
            help='encoder: cnn or bert')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-1, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
           help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
           help='sgd / adam / bert_adam')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--pair', action='store_true',
           help='use pair model')
    parser.add_argument('--eps', default=0.1, type=float,
           help='step size for SG-MCMC')
    parser.add_argument('--temp', default=10.0, type=float,
           help='temperature for softmax')
    parser.add_argument('--step', default=5, type=int,
           help='steps for SG-MCMC')
    parser.add_argument('--smp', default=10, type=int,
           help='samples for SG-MCMC')
    parser.add_argument('--ratio', default=0.01, type=float,
           help='decay ratio of step size for SG-MCMC')
    parser.add_argument('--wtp', default=0.1, type=float,
           help='weight of the prior term')
    parser.add_argument('--wtn', default=1.0, type=float,
           help='weight of the noise term')
    parser.add_argument('--wtb', default=0.0, type=float,
           help='weight of the background term')
    parser.add_argument('--metric', default='dot',
           help='similarity metric (dot or l2)')
    parser.add_argument('--seed', default=1234, type=int,
           help='random seed')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    
    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length)
    elif encoder_name == 'bert':
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                    './pretrain/bert-base-uncased',
                    max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                    './pretrain/bert-base-uncased',
                    max_length)
    else:
        raise NotImplementedError
    
    if opt.pair:
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader_pair(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    else:
        train_data_loader = get_loader(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        if opt.adv:
           adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'bert_adam':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv:
        d = Discriminator(opt.hidden_size)
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader, adv=opt.adv, d=d)
    else:
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    
    # Loading relation embeddings.
    data = pickle.load(open('./data/embeddings.pkl', 'rb'))
    rellist = data['relations']
    relemb = data['embeddings']
    array0 = np.zeros((1,relemb.shape[1]), dtype=relemb.dtype)
    relemb = np.concatenate([array0, relemb], axis=0)
    rel2id = dict([(rel, k + 1) for k, rel in enumerate(rellist)])
    # Loading relation graphs.
    with open('./data/graph.txt', 'r') as fi:
        us, vs, ws = [], [], []
        for line in fi:
            items = line.strip().split('\t')
            us += [rel2id[items[0]]]
            vs += [rel2id[items[1]]]
            ws += [float(items[2])]
        index = torch.LongTensor([us, vs])
        value = torch.Tensor(ws)
        shape = torch.Size([len(rel2id) + 1, len(rel2id) + 1])
        reladj = torch.sparse.FloatTensor(index, value, shape).cuda()
    # End
    model = REGRAB(sentence_encoder, hidden_size=opt.hidden_size, eps=opt.eps, temp=opt.temp, step=opt.step, smp=opt.smp, ratio=opt.ratio, wtp=opt.wtp, wtn=opt.wtn, wtb=opt.wtb, metric=opt.metric)
    model.set_relemb(rel2id, relemb)
    model.set_reladj(reladj)
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name == 'bert':
            bert_optim = True
        else:
            bert_optim = False

        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim)
    else:
        ckpt = opt.load_ckpt

    acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
    print("RESULT: %.2f" % (acc * 100))

if __name__ == "__main__":
    main()
