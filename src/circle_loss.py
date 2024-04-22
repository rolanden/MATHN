import torch
from torch import nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
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

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]
        #relu改成clamp?
        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss


if __name__ == '__main__':
    batch_size = 10
    feats = torch.rand(batch_size, 1028)
    labels = torch.randint(high=10, dtype=torch.long, size=(batch_size,))
    circleloss = CircleLoss(similarity='cos')
    print(circleloss(feats, labels))