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
    #print('标签',labels.size())
    '''
        #尝试同模自适应
    m=(dist_mat - (~(is_same_modal)) * 1000)
    #print(m)
    dist_samemodal_inds = torch.nonzero(m> 0)
    #print(dist_samemodal_inds.shape)
    #print(dist_samemodal_inds)
    q=torch.gather(dist_mat,dim=0,index=dist_samemodal_inds)
    #print(dist_mat.shape)
    #print(q.shape)
    q=q[:,1]
    #idx=torch.reshape(dist_samemodal_inds,(96,96))   #看看标签
    #print(idx)
    q=torch.reshape(q,(48,96))     #(-1,batchsize)

    #partial apc an
    m2 = (dist_mat - (is_pos&is_same_modal) * 1000)

    dist_samemodal_inds =torch.nonzero(m2>0)
    #print(dist_samemodal_inds)
    q2=torch.gather(dist_mat,dim=0,index=dist_samemodal_inds)
    q2=q2[:,1]
    q2=torch.reshape(q2,(92,96))



    #within apc anc
    m3=(dist_mat-(is_pos&~is_same_modal)*10000)
    dist_samemodal_inds = torch.nonzero(m3>0)
    q3=torch.gather(dist_mat,dim=0,index=dist_samemodal_inds)
    q3=q3[:,1]
    q3=torch.reshape(q3,(92,96))

    m4=(dist_mat-(is_pos^is_same_modal)*10000)   #减去异或为1的
    dist_samemodal_inds = torch.nonzero(m4>0)
    q4=torch.gather(dist_mat,dim=0,index=dist_samemodal_inds)
    q4=q4[:,1]
    q4=torch.reshape(q4,(48,96))


    #samemodalmargin = torch.mean(dim=1)
    #dist_mean = float(dist_mat.mean())
    #dist_std = float(dist_mat.std())
    dist_mean = dist_mat.mean(dim=0)
    dist_std = dist_mat.std(dim=0)

    #margin1
    hyper_mean1 = 0.3
    hyper_std1 = 100000

    #dist_mean1 = float(q.mean())
    #dist_std1 = float(q.std())
    dist_mean1 = q.mean(dim=0)
    #print(q)
    #print(dist_mean1)
    dist_std1 = q.std(dim=0)
    #print(dist_std1)
    dist_re = (hyper_std1 * (q - dist_mean1) / dist_std1 + hyper_mean1)
    #dist_mat= hyper_std * (dist_mat- dist_mean) / dist_std + hyper_mean
    dist_re=dist_re
    margin1 = dist_re.mean(dim=0)   #每个feather一个margin
    #margin1 = dist_re.mean()      #margin一个值
    #margin1 = torch.zeros(96).to('cuda:0')+margin1
    margin1=torch.clamp(margin1,0.01,0.8)
    torch.set_printoptions(precision=10)
    #print(margin1)

    #margin2 partial apc an
    hyper_mean2 = 0.3
    hyper_std2 = 100000
    dist_mean2 =q2.mean(dim=0)
    #print(dist_mean2)
    dist_std2 = q2.std(dim=0)
    dist_re = hyper_std2 *(q2 - dist_mean2) / dist_std2 +hyper_mean2
    margin2 = dist_re.mean(dim=0)
    margin2 = torch.clamp(margin2,0.01,0.8)
    #print(margin2)




    #margin3   within apc anc
    hyper_mean3 = 0.3
    hyper_std3 = 100000
    dist_mean3 = q3.mean(dim=0)
    dist_std3 =q3.std(dim=0)
    dist_re = hyper_std3 * (q3 - dist_mean3) / dist_std3 + hyper_mean3
    margin3 = dist_re.mean(dim=0)
    margin3 = torch.clamp(margin3,0.01,0.8)

    hyper_mean4 = 0.3
    hyper_std4 = 100000
    dist_mean4= q4.mean(dim=0)
    dist_std4 =q4.std(dim=0)
    dist_re = hyper_std4 * (q4 - dist_mean4) / dist_std4 + hyper_mean4
    margin4 = dist_re.mean(dim=0)
    margin4 = torch.clamp(margin4,0.01,0.8)

    

    margin114514 = torch.zeros(96) + 0.3
    margin114514 = margin114514.to('cuda:0')
    #margin1 = [margin1, margin114514.detach(), margin114514.detach()]
    #samemodalmargin = torch.cat(margin1).to('cuda:0')

    margin3 = [margin1.detach(), margin2.detach(), margin3.detach(),margin4.detach()]
    margin_mat = torch.cat(margin3).to('cuda:0')
    '''
    '''
    #用于算固定margin懒得改其他地方时候用
    margin114514 = torch.zeros(96) + 0.3
    margin114514 = margin114514.to('cuda:0')
    #margin1 = [margin1, margin114514.detach(), margin114514.detach()]
    #samemodalmargin = torch.cat(margin1).to('cuda:0')

    margin3 = [margin114514.detach(), margin114514.detach(),margin114514.detach()]
    margin_mat = torch.cat(margin3).to('cuda:0')
    '''



    if fix_margin=='nfix':
        #下面是用的ICCV那篇的margin的trick，然后把原来的margin矩阵改为了对每类求均值，获得一个margin。

        # 只要an距离（原文是ap，an都用了，我依稀记得全用结果会差点，上面那一坨注释了的好像就是全用的）
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

        #96*96的margintensor到底行不行呢
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

        margin114514 = torch.zeros(batch_size) + 0.3
        margin114514 = margin114514.to('cuda:0')


        '''
        #原来那些玩意
        margin3 = [margin1.detach(), margin2.detach(), margin3.detach()]
        # samemodalmargin = torch.cat(margin1).to('cuda:0')

        #margin3 = [margin2.detach(),margin3.detach()]
        margin_mat = torch.cat(margin3).to('cuda:0')

        '''

    if fix_margin == "fix":
        #首先得到内循环的margin，这里固定成margin
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

    '''

    #获得另一种距离，来计算
    #dist_ap2, relative_p_inds2 = torch.max(dist_mat2 - (is_neg + ~is_same_modal)*1000, 1, keepdim=True)
    #     1 0
    #     0 0    同模正图像  batchsize个
    #print(dist_ap)

    dist_an2, relative_n_inds2 = torch.min(dist_mat2 + (is_pos + ~is_same_modal) * 1000, 1, keepdim=True)
    #     1 1
    #     0 0    同模负图像
    #dist_ap2 = dist_ap2.squeeze(1)
    dist_an2 = dist_an2.squeeze(1)
    #dist_apc2, relative_pc_inds2 = torch.max(dist_mat2 - (is_neg + is_same_modal)*1000, 1, keepdim=True)
    #     1 0
    #     1 0    不同模正图像
    dist_anc2, relative_nc_inds2 = torch.min(dist_mat2 + (is_pos + is_same_modal) * 1000, 1, keepdim=True)
    #     1 0
    #     0 1    不同模负图像
    #dist_apc2 = dist_apc2.squeeze(1)
    dist_anc2 = dist_anc2.squeeze(1)

    '''

    #下面这段用的weekly那篇的方法
    '''

    beta1 = 0.3
    beta2 = 0.3
    beta3 = 0.3
    beta4 = 0
    margin_beta1 = beta1 + (dist_an-dist_ap ) / (3 - beta1)
    #margin1 = torch.zeros(96).to('cuda:0') + margin_beta1
    margin_beta2 = beta2 + (dist_anc-dist_apc) / (3 - beta2)
    #margin2 = torch.zeros(96).to('cuda:0') + margin_beta2
    margin_beta3 = beta3 + (dist_an-dist_apc ) / (3 - beta3)
    #margin3 = torch.zeros(96).to('cuda:0') + margin_beta3
    margin_beta4 = beta4 + (dist_anc - dist_ap) / (7 - beta4)
    margin3 = [margin_beta1.detach(),margin_beta2.detach(), margin_beta3.detach()]

    margin_mat = torch.cat(margin3).to('cuda:0')
    margin_mat = torch.clamp(margin_mat, 0.01, 0.8)
    margin_beta1 = torch.clamp(margin_beta1,0.01,0.8)
    #print(margin_mat)

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
        dist_ap = [dist_ap, dist_apc] #[dist_ap, dist_apc, dist_apc]
        dist_an = [dist_an, dist_an] #[dist_an., dist_an, dist_anc]
        dist_mix = [dist_anc1, dist_anc1]
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


class CrossMatchingSphereLoss(nn.Module):
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
            #print('edis',dist)
            max = torch.max(dist)
            #print('max',max)
            min = torch.min(dist)
            #print('min',min)
            dist = (dist-min)/(max-min)
            #print('dist',dist)

        dist2=euclidean_dist(feat1,feat2)
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
        sim_ap = torch.neg(dist_ap.type(torch.HalfTensor)).add(1).cuda()
        sim_an = torch.neg(dist_an.type(torch.HalfTensor)).add(1).cuda()
        sim_an2 = torch.neg(dist_mix.type(torch.HalfTensor)).add(1).cuda()
        scale = 32
        margin = 0.3
        O_p = - margin
        deta_p = margin
        O_n = 1+margin
        deta_n = 1-margin
        O_n2 = O_n
        deta_n2 = deta_n

        alpha_p = torch.relu(-sim_ap + O_p)
        alpha_n = torch.relu(sim_an - O_n)
        alpha_n2 = torch.relu(sim_an2 - O_n2)


        loss_p = torch.exp(-scale * alpha_p * (sim_ap - deta_p))
        loss_n = torch.exp(scale * alpha_n * (sim_an - deta_n))
        loss_n2 = torch.exp(scale * alpha_n2 * (sim_an2 - deta_n2))

        loss = torch.log(1 + loss_p * loss_n * loss_n2)

        '''
        within_margin=0.3
        within_loss = torch.relu(dist_ap_within - dist_anc_within + margin_beta1)
        within_loss = within_loss.mean(-1)
        '''
        return margin1,loss.mean(-1).sum(), dist_ap, dist_an,dist_mix,dist_mat#emd_loss.mean(-1).sum()


class WeightedCrossMatchingSphereLoss(CrossMatchingSphereLoss):

    beta = 30

    def forward(self, global_feat, labels, tags,fix_margin, fmargin):
        assert self.mode in ['easy', 'all']
        #_, dist_ap, dist_an = super().forward(global_feat, labels, tags)
        margin1,_, dist_ap, dist_an,dist_mix,dist_mat= super().forward(global_feat, labels, tags,fix_margin, fmargin)
        #margin1=margin1.to('cuda:0')

        #print(dist_ap.size())
        #print(dist_ap)
        #print(margin1)
        '''
        #相似度版本
        sim_ap = torch.neg(dist_ap.type(torch.HalfTensor)).add(1).cuda()
        sim_an = torch.neg(dist_an.type(torch.HalfTensor)).add(1).cuda()
        sim_an2 = torch.neg(dist_mix.type(torch.HalfTensor)).add(1).cuda()
        #print('sim',sim_ap)
        scale = 32
        margin = 0.3
        O_p = 1+margin
        deta_p = 1-margin
        O_n = -margin
        deta_n = margin
        O_n2 = O_n
        deta_n2 = deta_n

        alpha_p = torch.relu(-sim_ap +O_p)
        alpha_n = torch.relu(sim_an -O_n)
        alpha_n2 = torch.relu(sim_an2 - O_n2)



        loss_p = torch.exp(-scale * alpha_p * (sim_ap - deta_p))
        loss_n = torch.exp(scale * alpha_n * (sim_an - deta_n))
        loss_n2 = torch.exp(scale * alpha_n2 * (sim_an2 - deta_n2))

        #loss = torch.log(1 + (loss_p * loss_n*loss_n2).sum()) #1号，直接加
        # loss = torch.log(1 + loss_p * loss_n*loss_n2) #2号，后面再加
        loss = torch.log(1 + loss_p * loss_n)


        

        '''
        #距离版本 7.429 7.07 8.07

        scale = 32
        margin = 0.3
        O_p =  0.2-margin
        deta_p =  0.2+margin
        O_n = 0.9+margin
        deta_n = 0.9-margin
        O_n2 = O_n
        deta_n2 = deta_n

        alpha_p = torch.relu(dist_ap - O_p)
        alpha_n = torch.relu(-dist_an + O_n)
        alpha_n2 = torch.relu(-dist_mix + O_n2)

        loss_p = torch.exp(scale * alpha_p * (dist_ap - deta_p))
        loss_n = torch.exp(-scale * alpha_n * (dist_an - deta_n))
        loss_n2 = torch.exp(-scale * alpha_n2 * (dist_mix - deta_n2))

        # loss = torch.log(1 + (loss_p * loss_n*loss_n2).sum()) #1号，直接加
        #loss = torch.log(1 + loss_p * loss_n*loss_n2) #2号，后面再加
        loss = torch.log(1 + loss_p * loss_n)






        #print('loss_p',loss_p)
        '''
        #新三元组
        alpha = 0.3
        dist1 = dist_ap/(1+torch.exp(-dist_ap+margin1))
        dist2 = torch.clamp_max(margin1-dist_an,0)/(1+torch.exp(dist_an-margin1))
        dist=alpha*dist1+(1-alpha)*dist2
        '''

        #dist = dist_ap - dist_an + self.margin

        #print(within_loss)



        dist = loss.reshape(-1, labels.shape[0])

        # soft relu for better gradient estimation
        loss = F.softplus(dist, beta=self.beta).mean(-1)

        # gradient for softplus
        loss_grad = torch.sigmoid(dist*self.beta).mean(-1)

        # gradient-based weighting
        weight = loss_grad.mean() / loss_grad.clamp_min(0.1)
        #print(weight)
        loss = loss * weight[:, None]

        #weight = torch.tensor([1.07,0.96,0.97]).to('cuda:0').detach()
        loss = loss * weight
        #print(loss.size())
        loss = loss.mean(-1).sum()
        #loss = 0.65*loss+0.35*within_loss
        #print(loss)
        # loss = loss.sum()
        #print('loss',loss)




        emd_loss=torch.zeros([1,2])
        return dist_mix,loss, dist_ap, dist_an,dist_mat#原本就是emd_loss


