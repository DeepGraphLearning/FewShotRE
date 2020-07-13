import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class REGRAB(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230, eps=0.1, temp=100.0, step=10, smp=1, ratio=1.0, wtp=1.0, wtn=1.0, wtb=1.0, metric='dot'):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.eps = eps
        self.temp = temp
        self.step = step
        self.smp = smp
        self.ratio = ratio
        self.wtp = wtp
        self.wtn = wtn
        self.wtb = wtb
        self.metric = metric
        print('---------- Hyperparameters ----------')
        print('Epsilon: {}'.format(self.eps))
        print('Temperature: {}'.format(self.temp))
        print('Steps: {}'.format(self.step))
        print('Samples: {}'.format(self.smp))
        print('Decay Ratio: {}'.format(self.ratio))
        print('Weight of Prior: {}'.format(self.wtp))
        print('Weight of Noise: {}'.format(self.wtn))
        print('Weight of Background: {}'.format(self.wtb))
        print('Similarity Metric: {}'.format(self.metric))
        print('-------------------------------------')
        self.fc = nn.Linear(512, hidden_size * 2)
        self.drop = nn.Dropout()

    def set_relemb(self, rel2id, relemb):
        self.rel2id = rel2id
        self.relemb = torch.tensor(relemb).cuda()
    
    def set_reladj(self, reladj):
        self.reladj = reladj
    
    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def __get_emb__(self, embs, pos):
        idx = pos.view(-1, 2, 2)[:, :, 0]
        idx = idx.unsqueeze(-1).expand(-1, -1, embs.size(-1))
        emb = torch.gather(embs, dim=1, index=idx).view(embs.size(0), -1)
        return emb
    
    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        proto_vec_prior = torch.spmm(self.reladj, self.relemb)
        proto_vec_prior = torch.tanh(self.fc(proto_vec_prior))

        relation_id = [self.rel2id.get(relation, 0) for relation in support['rel']]
        relation_id = torch.LongTensor(relation_id).cuda().view(-1, N, K)[:, :, 0]
        
        proto_vec_prior = proto_vec_prior[relation_id].unsqueeze(1).expand(-1, self.smp, -1, -1)

        support_emb = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        support_emb = self.__get_emb__(support_emb, support['pos'])
        
        query_emb = self.sentence_encoder(query) # (B * total_Q, D)
        query_emb = self.__get_emb__(query_emb, query['pos'])

        D = support_emb.size(-1)

        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)
        support_emb = support_emb.view(-1, N, K, D) # (B, N, K, D)
        query_emb = query_emb.view(-1, total_Q, D) # (B, total_Q, D)

        B = support_emb.size(0) # Batch size
        proto_vec_background = torch.mean(support_emb.view(B, -1, D), dim=1).unsqueeze(1).unsqueeze(1).expand(-1, self.smp, N, -1).detach()
         
        # Prototypical Networks 
        # Ignore NA policy
        proto_vec = torch.mean(support_emb, 2).unsqueeze(1).expand(-1, self.smp, -1, -1).detach() # (B, S, N, D)
        proto_vec = proto_vec + self.wtp * proto_vec_prior - self.wtb * proto_vec_background
        support_emb_expand = support_emb.unsqueeze(3).unsqueeze(1).expand(-1, self.smp, -1, -1, N, -1) # (B, S, N, K, N, D)
        eps = self.eps
        for k in range(self.step):
            proto_vec_expand = proto_vec.unsqueeze(2).unsqueeze(2).expand_as(support_emb_expand) # (B, S, N, K, N, D)
            if self.metric == 'dot':
                prob = torch.sum(support_emb_expand * proto_vec_expand, dim=-1)
            if self.metric == 'l2':
                prob = 0.5 * torch.sum(-torch.pow(support_emb_expand - proto_vec_expand, 2), dim=-1)
            prob = torch.softmax(prob / self.temp, dim=-1) # (B, S, N, K, N)
            eye = torch.eye(N).to(prob).unsqueeze(0).unsqueeze(0).unsqueeze(3).expand_as(prob)
            weight = eye - prob # (B, S, N, K, N)
            weight = weight.unsqueeze(-1) # (B, S, N, K, N, 1)
            support_emb_expand_ = support_emb.unsqueeze(1).unsqueeze(4).expand(-1, self.smp, -1, -1, -1, -1) # (B, S, N, K, 1, D)

            offset = weight * support_emb_expand_
            offset = offset.mean(2).mean(2)

            noise = torch.randn(proto_vec.size()).to(proto_vec) * np.sqrt(eps)
            proto_vec = proto_vec + 0.5 * eps * self.wtp / self.wtn * (proto_vec_prior - proto_vec) + 0.5 * eps / self.wtn * offset.detach() + noise
            eps = eps * self.ratio
        
        # (B, S, N, D)
        
        proto_vec = proto_vec.unsqueeze(2).expand(-1, -1, total_Q, -1, -1) # (B, S, total_Q, N, D)
        query_emb = query_emb.unsqueeze(1).unsqueeze(3).expand(-1, self.smp, -1, N, -1) # (B, S, total_Q, N, D)

        if self.metric == 'dot':
            logits = torch.sum(proto_vec * query_emb, dim=-1)
        if self.metric == 'l2':
            logits = 0.5 * torch.sum(-torch.pow(proto_vec - query_emb, 2), dim=-1)
        # ----------
        probs = torch.softmax(logits, dim=-1).mean(1) 
        #probs = logits.mean(1)
        # ----------
        logits = torch.log_softmax(logits, dim=-1).mean(1)

        _, pred = torch.max(probs.view(-1, N), 1)
        return logits, pred

    
    
    
