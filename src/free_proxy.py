import torch
from torch import nn
import torch.nn.functional as F



class proxyfree(nn.Module):
    def __init__(self, alpha=0.01, r=1, m=0, bias=0.3, similarity='euc', **kwargs):
        super(proxyfree, self).__init__()
        self.alpha= alpha
        self.r = r
        self.m = m
        self.bias = bias
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        m = labels.size(0)
        mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        elif self.similarity== 'euc':
            feats = F.normalize(feats)
            m, n = feats.size(0), feats.size(0)
            xx = torch.pow(feats, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(feats, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(1, -2, feats, feats.t())  #
            sim_mat = dist.clamp(min=1e-12).sqrt()  # for numerical stability

            #为啥对角线还。。
            diag = torch.diag(sim_mat)
            diag = torch.diag_embed(diag)
            sim_mat = sim_mat - diag


            max = torch.max(sim_mat)
            min = torch.min(sim_mat)
            sim_mat = (sim_mat-min)/(max-min) #归一化然后负数+1，变为相似度
            sim_mat = torch.neg(sim_mat.type(torch.HalfTensor)).add(1).cuda()

            #print(sim_mat)
            #print(sim_mat.dtype)
            # print(dist.size())
        else:
            raise ValueError('This similarity is not implemented.')




        '''
        y_label = torch.eq(labels.t(),labels)
        mask_p = y_label
        print('p',mask_p)
        y_label = torch.logical_not(y_label)
        #同类为1，不同类0
        mask_n= y_label
        print('n',mask_n)
        '''

        mask_p = labels.view(-1, 1).eq(labels.view(1, -1))
        mask_n = mask_p.logical_not()

        mask_p.fill_diagonal_(False)
        #print('pp',mask_p)

        logits_p = torch.masked_select(sim_mat, mask_p)
        #print('p',logits_p.size())
        logits_p = (logits_p - self.m + self.bias) / self.r
        logits_n = torch.masked_select(sim_mat, mask_n)
        #print('n', logits_n.size())
        logits_n = (logits_n + self.m + self.bias) * self.r

        # loss
        loss_p = F.binary_cross_entropy_with_logits(logits_p, torch.ones_like(logits_p))
        loss_n = F.binary_cross_entropy_with_logits(logits_n, torch.zeros_like(logits_n))
        loss = self.alpha * loss_p + (1. - self.alpha) * loss_n

        #print('loss:',loss)



        return loss
