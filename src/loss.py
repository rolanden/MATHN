import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
from emd import SinkhornDistance
import torch.utils.data
from utils import save_checkpoint, resume_from_checkpoint, \
    load_data, build_model_optm, get_train_args, fix_base_para,get_new_params

global args
    # args = parser.parse_args()
args = get_train_args()
batch_size=args.batch_size



def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())       #
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    #print(dist.size())
    return dist

def cos_dist(x,y):
    m, n = x.size(0), y.size(0)
    xx=F.normalize(x,1).to('cuda:0')
    yy=F.normalize(y,1).t().to('cuda:0')
    q=torch.zeros([m,n]).to('cuda:0')
    dist=q.addmm_(0,0,xx,yy).to('cuda:0')
    dist=dist.clamp(min=1e-12).sqrt()
    return dist




def pairwise_mahalanobis(S1, S2, Cov_1=None):
    """
        S1: C1 x K matrix (torch.FloatTensor)
          -> C1 K-dimensional semantic prototypes
        S2: C2 x K matrix (torch.FloatTensor)
          -> C2 K-dimensional semantic prototypes
        Sigma_1: K x K matrix (torch.FloatTensor)
          -> inverse of the covariance matrix Sigma; used to compute Mahalanobis distances
          by default Sigma is the identity matrix (and so distances are euclidean distances)

        returns an C1 x C2 matrix corresponding to the Mahalanobis distance between each element of S1 and S2

    """
    if S1.dim() != 2 or S2.dim() != 2 or S1.shape[1] != S2.shape[1]:
        raise RuntimeError("Bad input dimension")
    C1, K = S1.shape
    C2, K = S2.shape
    if Cov_1 is None:
        Cov_1 = torch.eye(K).to('cuda:0')
    if Cov_1.shape != (K, K):
        raise RuntimeError("Bad input dimension")

    S1S2t = S1.matmul(Cov_1).matmul(S2.t()).to('cuda:0')
    S1S1 = S1.matmul(Cov_1).mul(S1).sum(dim=1, keepdim=True).expand(-1, C2).to('cuda:0')
    S2S2 = S2.matmul(Cov_1).mul(S2).sum(dim=1, keepdim=True).t().expand(C1, -1).to('cuda:0')
    return torch.sqrt(torch.abs(S1S1 + S2S2 - 2. * S1S2t) + 1e-32).to('cuda:0') # to avoid numerical instabilities


def cross_modal_hard_example_mining(dist_mat,fix_margin,fmargin, labels, tags, mode='all', return_ind=False):

    assert len(dist_mat.size()) == 2


    #pos和neg判断？

    is_pos = labels[:, None].eq(labels)
    is_neg = labels[:, None].ne(labels)   #pos和neg是反着的，一个true一个就false，pos true少，neg多
    is_same_modal = tags[:, None].eq(tags)    #是否同模态
    N, M = dist_mat.shape
    torch.set_printoptions(threshold=np.inf)  #让tensor显示全
    #print('neg')
    #print(is_neg)
    #print('pos')
    #print(is_pos)
    #print(is_neg&is_pos)
    #print('model')
    #print(is_same_modal)
    #print('标签',labels)


    #print('fix margin?',fix_margin)
    if fix_margin=='unfix':


        # 只要an距离（我依稀记得全用结果会差点，上面那一坨注释了的好像就是全用的）
        m = (dist_mat - (is_pos|~is_same_modal) * 1000)
        # print(m)
        dist_samemodal_inds = torch.nonzero(m > 0)
        # print(dist_samemodal_inds.shape)
        # print(dist_samemodal_inds)
        q = torch.gather(dist_mat, dim=0, index=dist_samemodal_inds)
        # print(dist_mat.shape)
        # print(q.shape)
        q = q[:, 1]
        # idx=torch.reshape(dist_samemodal_inds,(96,96))   #看看标签
        # print(idx)
        q = torch.reshape(q, (-1, batch_size))  # (-1,batchsize)

        # partial apc an
        m2 = (dist_mat - (is_pos|~is_same_modal) * 1000)

        dist_samemodal_inds = torch.nonzero(m2 > 0)
        # print(dist_samemodal_inds)
        q2 = torch.gather(dist_mat, dim=0, index=dist_samemodal_inds)
        q2 = q2[:, 1]
        q2 = torch.reshape(q2, (-1, batch_size))

        # within apc anc
        m3 = (dist_mat - (is_pos |is_same_modal) * 10000)
        dist_samemodal_inds = torch.nonzero(m3 > 0)
        q3 = torch.gather(dist_mat, dim=0, index=dist_samemodal_inds)
        q3 = q3[:, 1]
        q3 = torch.reshape(q3, (-1, batch_size))

        # samemodalmargin = torch.mean(dim=1)
        # dist_mean = float(dist_mat.mean())
        # dist_std = float(dist_mat.std())
        dist_mean = dist_mat.mean(dim=0)
        dist_std = dist_mat.std(dim=0)

        # margin1
        hyper_mean1 = 0.3
        hyper_std1 = 75000

        # dist_mean1 = float(q.mean())
        # dist_std1 = float(q.std())
        dist_mean1 = q.mean(dim=0)
        # print(q)
        # print(dist_mean1)
        dist_std1 = q.std(dim=0)
        # print(dist_std1)
        dist_re = (hyper_std1 * (q - dist_mean1) / dist_std1 + hyper_mean1)
        # dist_mat= hyper_std * (dist_mat- dist_mean) / dist_std + hyper_mean
        dist_re = dist_re
        margin1 = dist_re.mean(dim=0)  # 每个feather一个margin
        # margin1 = dist_re.mean()      #margin一个值
        # margin1 = torch.zeros(96).to('cuda:0')+margin1

        #
        #margin1 = dist_re
        #
        margin1 = torch.relu(margin1)



        torch.set_printoptions(precision=10)
        # print(margin1)

        # margin2 partial apc an
        hyper_mean2 = 0.3
        hyper_std2 = 75000
        dist_mean2 = q2.mean(dim=0)
        # print(dist_mean2)
        dist_std2 = q2.std(dim=0)
        dist_re = hyper_std2 * (q2 - dist_mean2) / dist_std2 + hyper_mean2
        margin2 = dist_re.mean(dim=0)

        #
        #margin2 = dist_re
        #
        margin2 = torch.relu(margin2)

        # print(margin2)

        # margin3   within apc anc
        hyper_mean3 = 0.3
        hyper_std3 = 75000
        dist_mean3 = q3.mean(dim=0)
        dist_std3 = q3.std(dim=0)
        dist_re = hyper_std3 * (q3 - dist_mean3) / dist_std3 + hyper_mean3
        margin3 = dist_re.mean(dim=0)

        #
        #margin3= dist_re
        #
        margin3 = torch.relu(margin3)







        #固定的margin，要想用固定的margin就把下面的margin3里三个全改成114514
        '''
        margin114514 = torch.zeros(batch_size) + 0.3
        margin114514 = margin114514.to('cuda:0')


        '''
        #原来那些玩意
        margin3 = [margin1.detach(), margin2.detach(), margin3.detach()]
        # samemodalmargin = torch.cat(margin1).to('cuda:0')

        #margin3 = [margin2.detach(),margin3.detach()]
        margin_mat = torch.cat(margin3).to('cuda:0')



    elif fix_margin == "fix":
        #print("fixed margin")
        #print(args.fix_margin)

        margin_mat = fmargin


    dist_ap, relative_p_inds = torch.max(dist_mat - (is_neg + ~is_same_modal)*1000, 1, keepdim=True)
    #     1 0
    #     0 0    同模正图像  batchsize个

    #print(dist_ap)

    dist_an, relative_n_inds = torch.min(dist_mat + (is_pos + ~is_same_modal) * 1000, 1, keepdim=True)
    #     1 1
    #     0 0    同模负图像
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    dist_apc, relative_pc_inds = torch.max(dist_mat - (is_neg + is_same_modal)*1000, 1, keepdim=True)
    #     1 0
    #     1 0    不同模正图像
    dist_anc, relative_nc_inds = torch.min(dist_mat + (is_pos + is_same_modal) * 1000, 1, keepdim=True)
    #     1 0
    #     0 1    不同模负图像
    dist_apc = dist_apc.squeeze(1)
    dist_anc = dist_anc.squeeze(1)


    '''单向,消融用'''
    '''
    tag_len = len(tags)

    tag2 = torch.unsqueeze(tags, 0).repeat(tag_len, 1)
    #print('tag222222',tag2)
    #print('distmat',dist_mat)

    dist_ap, relative_p_inds = torch.max(dist_mat - (is_neg + ~is_same_modal)*1000-tag2*1000, 1, keepdim=True)
    #     1 0
    #     0 0    同模正图像  batchsize个

    #print(dist_ap)

    dist_an, relative_n_inds = torch.min(dist_mat + (is_pos + ~is_same_modal) * 1000+tag2*1000, 1, keepdim=True)
    #     1 1
    #     0 0    同模负图像
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    dist_apc, relative_pc_inds = torch.max(dist_mat - (is_neg + is_same_modal)*1000-tag2*1000, 1, keepdim=True)
    #     1 0
    #     1 0    不同模正图像
    dist_anc, relative_nc_inds = torch.min(dist_mat + (is_pos + is_same_modal) * 1000+tag2*1000, 1, keepdim=True)
    #     1 0
    #     0 1    不同模负图像
    dist_apc = dist_apc.squeeze(1)
    dist_anc = dist_anc.squeeze(1)

    #print('dis',dist_ap)

    '''






    margin_beta1= 0
    emd_loss = dist_ap+dist_apc-dist_an-dist_anc+0.3
    #print(dist_ap.size(),dist_apc.size(), dist_apc.size())
    #print('ap{}\tan{}\tapc{}\tanc{}'.format(dist_ap,dist_an,dist_apc,dist_anc))

    #dist_ap1 = [dist_ap.detach(),dist_apc.detach()]
    #dist_an1 = [dist_an.detach(),dist_anc.detach()]
    #要非同模的就行了，或者换着展示
    dist_ap1 = dist_ap
    dist_an1 = dist_an
    dist_apc1 = dist_apc
    dist_anc1 = dist_anc
    # the detached term is added to make sure the gradient of each loss won't change when combining multiple losses
    #cross   我感觉是within
    if mode == 'basic':
        dist_ap = [dist_ap, dist_ap.detach(), dist_ap.detach()]
        dist_an = [dist_an, dist_an.detach(), dist_an.detach()]
    #within   我感觉是cross
    elif mode == 'within':
        dist_ap = [dist_apc, dist_apc.detach(), dist_apc.detach()]
        dist_an = [dist_anc, dist_anc.detach(), dist_anc.detach()]
    #hybird   感觉是对的
    elif mode == 'partial':
        dist_ap = [dist_apc, dist_apc.detach(), dist_apc.detach()]
        dist_an = [dist_an, dist_an.detach(), dist_an.detach()]
    #mathm   [basic  partial within]
    elif mode == 'all':
        dist_ap = [dist_ap, dist_apc, dist_apc] #[dist_ap, dist_apc, dist_apc]
        dist_an = [dist_an, dist_an, dist_anc] #[dist_an., dist_an, dist_anc]
        dist_mix = [dist_anc1, dist_anc1, dist_an1.detach()]
        dist_mix = torch.cat(dist_mix)
    elif mode == 'easy':
        dist_ap = [dist_ap, dist_apc.detach(), dist_apc]  #[dist_ap, dist_apc, dist_apc]
        dist_an = [dist_an, dist_an.detach(), dist_anc]  #[dist_an., dist_an, dist_anc]


    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)




    if return_ind is False:
        return margin_mat,dist_ap, dist_an,dist_ap1, dist_an1,dist_mix#emd_loss
    else:
        pind = torch.cat([relative_p_inds, relative_pc_inds, relative_p_inds, relative_pc_inds, ])
        nind = torch.cat([relative_n_inds, relative_n_inds, relative_nc_inds, relative_nc_inds, ])
        return margin_mat,dist_ap, dist_an, pind, nind


class CrossMatchingTripletLoss(nn.Module):
    def __init__(self, margin=0, normalize_feature=False, mode='all', share_neg=True):
        super().__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        # assert mode in ['all', 'same', 'cross']
        assert mode in ['basic', 'within', 'partial','easy', 'all']
        self.mode = mode
        self.share_neg = share_neg


    @staticmethod

    def get_dist(feat1, feat2):
        flag=0 #youwenti
        if flag==0:
            dist=euclidean_dist(feat1,feat2)
        elif flag ==1:
            dist=1-cos_dist(feat1,feat2)
        elif flag == 2:
            dist=pairwise_mahalanobis(feat1,feat2)
        elif flag == 3:
            emd = SinkhornDistance(eps=0.1, max_iter=100)
            dist2, P, C = emd(feat1, feat2)
            #dist2 = wasserstein_distance(feat1.cpu(),feat2.cpu())
        elif flag == 4:
            dist=euclidean_dist(feat1,feat2)
            max = torch.max(dist)
            min = torch.min(dist)
            dist = (dist-min)/(max-min)




        dist2= None
        return dist,dist2

    def forward(self, global_feat, labels, tags,fix_margin, fmargin):
        #print("挖掘",fix_margin)
        if self.normalize_feature:
            global_feat = F.normalize(global_feat, dim=-1)
        dist_mat,dist_mat2 = self.get_dist(global_feat, global_feat)
        #print(dist_mat.size())
        '''
        #尝试

        dist_mean = float(dist_mat.mean())
        dist_std = float(dist_mat.std())
        hyper_mean = 1.35
        hyper_std = 0.16
        dist_re = hyper_std * (dist_mat - dist_mean) / dist_std + hyper_mean-1
        #print(dist_mean)
        #print(dist_std)
        if self.mode=='basic':
            margin1 = dist_re.mean(dim=1)
            margin1 = [margin1, margin1.detach(), margin1.detach()]
            margin1 = torch.cat(margin1).to('cuda:0')
        elif self.mode == 'all':
            margin1 = torch.zeros(96)+0.2
            margin1 = [2*margin1, 2*margin1, 0.75*margin1]
            margin1 = torch.cat(margin1).to('cuda:0')
        '''

        #margin1 = dist_re.mean()

        #dist_ap, dist_an = cross_modal_hard_example_mining(dist_re, labels, tags, self.mode)
        margin1,dist_ap, dist_an,dist_ap_within,dist_anc_within, dist_mix= cross_modal_hard_example_mining(
            dist_mat,fix_margin,fmargin, labels, tags, self.mode)

        '''
        _a,_b, _c,_d,_e, emd_loss = cross_modal_hard_example_mining(
            dist_mat2,fix_margin,fmargin, labels, tags, self.mode)
        '''
        #emd_loss=torch.zeros([1,2])
        #loss = torch.relu(dist_ap - dist_an + self.margin)
        #margin1=0.3
        loss = torch.relu(dist_ap - dist_an + margin1)

        '''
        within_margin=0.3
        within_loss = torch.relu(dist_ap_within - dist_anc_within + margin_beta1)
        within_loss = within_loss.mean(-1)
        '''
        return margin1,loss.mean(-1).sum(), dist_ap, dist_an,dist_mix,dist_mat#emd_loss.mean(-1).sum()


class WeightedCrossMatchingTripletLoss(CrossMatchingTripletLoss):

    beta = 30

    def forward(self, global_feat, labels, tags,fix_margin, fmargin):
        assert self.mode in ['easy', 'all']
        #_, dist_ap, dist_an = super().forward(global_feat, labels, tags)
        margin1,_, dist_ap, dist_an,dist_mix,dist_mat= super().forward(global_feat, labels, tags,fix_margin, fmargin)
        #margin1=margin1.to('cuda:0')

        #print(dist_ap.size())
        #print(dist_ap)
        #print(margin1)
        #margin1=0.3

        #dist = dist_ap - 0.5*dist_an -0.5*dist_mix+ margin1##?>?

        '''
        #新三元组
        alpha = 0.3
        dist1 = dist_ap/(1+torch.exp(-dist_ap+margin1))
        dist2 = torch.clamp_max(margin1-dist_an,0)/(1+torch.exp(dist_an-margin1))
        dist=alpha*dist1+(1-alpha)*dist2
        '''

        dist = dist_ap - dist_an + margin1

        #print(within_loss)

        dist = dist.reshape(-1, labels.shape[0])

        # soft relu for better gradient estimation
        loss = F.softplus(dist, beta=self.beta).mean(-1)

        # gradient for softplus
        loss_grad = torch.sigmoid(dist*self.beta).mean(-1)

        # gradient-based weighting
        weight = loss_grad.mean() / loss_grad.clamp_min(0.1)
        #print(weight)
        '''
        权重改掉
        
        
        loss = loss * weight[:, None]

        #weight = torch.tensor([1.07,0.96,0.97]).to('cuda:0').detach()
        loss = loss * weight
        '''
        #print(loss.size())
        loss = loss.mean(-1).sum()
        #loss = 0.65*loss+0.35*within_loss
        #print(loss)
        # loss = loss.sum()


        emd_loss=torch.zeros([1,2])
        return margin1,loss, dist_ap, dist_an,dist_mat#原本就是emd_loss


