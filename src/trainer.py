import torch
import torch.nn as nn
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from circle_loss import CircleLoss
from apex import amp
from utils import accuracy, save_checkpoint, AverageMeter,fix_base_para,get_new_params
import matplotlib.pyplot as plt

import torch.nn.functional as F


from torch.autograd import Variable
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from utils import  get_train_args
global args
    # args = parser.parse_args()
args = get_train_args()
batch_size=args.batch_size


















def train(train_loader, model,  criterion, criterion_train_t, \
               optimizer, epoch, args, fix_margin,fmargin,criterion_train_c,\
          model2,optimizer2,loss_kd1,loss_kd2):

    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_t = AverageMeter()
    losses1 = AverageMeter()
    avg_s_ap = AverageMeter()
    avg_s_an = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    s_ap = torch.tensor(0., device=device)
    s_an = torch.tensor(0., device=device)


    losses_kld=AverageMeter()
    losses_kd_1=AverageMeter()
    losses_kd_2=AverageMeter()







    # switch to train mode
    model.train()
    end = time.time()
    margin_layer = ['linear1','linear2', 'linear3', 'linear4','linear5']
    print(type(train_loader))
    for i, (input_all, target_all, tag_all) in enumerate(train_loader):
        input_all = input_all.to(device)
        tag_all = tag_all.to(device)
        target_all = target_all.type(torch.LongTensor).view(-1,).to(device)
        '''
        # 将这些数据转换成Variable类型
        inputs1, labels1 ,target= Variable(input_all), Variable(tag_all),Variable(target_all)
        # 接下来就是跑模型的环节了，我们这里使用print来代替
        print("epoch：", epoch, "的第", i, "个inputs", inputs1.data.size(), "labels", labels1.data,'target',target.data)
        '''

        if epoch>=0:

            for name, param in model.named_parameters():
                # Set True only for params in the list 'params_to_train'model.named_parameters()
                #param.requires_grad = True if name in params_not_to_train else False
                for ml in margin_layer:
                    #print(ml)
                    if ml in name:
                        param.requires_grad = True
        elif epoch<0:
            for name, param in model.named_parameters():
                # Set True only for params in the list 'params_to_train'model.named_parameters()
                #param.requires_grad = True if name in params_not_to_train else False

                for ml in margin_layer:
                    #print(ml)
                    if ml in name:
                        param.requires_grad = False

        optimizer.zero_grad()


        #model.requires_grad_(True)

        output, feat,fmargin = model(input_all, tag_all, return_feat=True)

        #那个对齐的损失
        loss_kld = torch.Tensor([0]).cuda()
        zn = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (len(input_all), feat.size(-1)))))
        loss_kld = F.kl_div(F.log_softmax(feat, dim=-1), F.softmax(zn, dim=-1), reduction='batchmean')
        print_margin=fmargin


        '''
        #新加
        min,_=torch.min(fmargin,dim=0)
        #print(min)
        max,_=torch.max(fmargin,dim=0)
        fmargin=(fmargin-min)/(max-min)
        fmargin = torch.mean(fmargin,dim=0)

        
        print('hh',fmargin.size())  #96 3
        fmargin = torch.norm(fmargin, p=1, dim=0)
        #print('hhs', fmargin.size())  #3
        '''
        #fmargin = fmargin.repeat(batch_size,1).transpose(dim0=1,dim1=0) #修改1为3确保size，或者直接不管模型输出那边的3.。。反正detach
        fmargin = fmargin.repeat(batch_size, 1).transpose(dim0=1, dim1=0)
        #print(fmargin)
        #print(fmargin.size())
        #print(margin)
        #print('1',margin.size())
        fmargin = fmargin.clamp(min=1e-12).reshape(-1,1).squeeze(1)



        fmargin =Variable(fmargin,requires_grad =True)
        #print(fmargin.size())
        #print(output.size())#[96 220] class
        #print(feat.size()) #[96 512]


        #
        loss = criterion(output, target_all)
        #更换成feat用于free代理
        #loss = criterion(feat, target_all)

        #loss_c = criterion_train_c(feat, target_all)

        #print(fmargin)
        #在这里do一次内循环更新
        if epoch <0:
            fmargin=0.3
        #loss1 = criterion_train_c(feat, target_all)  #circle
        #loss1 = criterion_train_c(feat, target_all, tag_all,fix_margin,fmargin) #sphere
        # local metric loss
        if args.tri_lambda > 0:
            s_mix,loss_t, s_ap, s_an,dist_mat = criterion_train_t(feat, target_all, tag_all,fix_margin,fmargin)
            #target是label,tag是模态
        else:
            loss_t = 0 * loss
        '''用proxy替代了tri loss
        '''
        #loss_t=loss_c

        '''
        距离图
        '''
        #distance_view(target_all, tag_all, dist_mat)
        #distance_view2(target_all, tag_all, dist_mat)
        #distance_view3(target_all, tag_all, dist_mat)

        if epoch <= 5:
            model2.eval()
            with torch.no_grad():
                output_t,feat_t,fmargin_t = model2(input_all, tag_all, return_feat=True)
            loss_kd_1 = loss_kd1(output, output_t)#最后一个先none

            #print('?',output_t.size(),output.size(),target_all.size())
            #loss_kd_1 = nn.CrossEntropyLoss(output, output_t)


            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target_all, topk=(1, 5))
            losses.update(loss.item(), input_all.size(0))
            losses_t.update(loss_t.item(), input_all.size(0))
            losses_t.update(loss_t.item(), input_all.size(0))
            avg_s_ap.update(s_ap.mean().item(), input_all.size(0))
            avg_s_an.update(s_an.mean().item(), input_all.size(0))
            top1.update(acc1[0], input_all.size(0))
            top5.update(acc5[0], input_all.size(0))

            # kld
            losses_kld.update(loss_kld.item(), input_all.size(0))
            losses_kd_1.update(loss_kd_1.item(), input_all.size(0))






            # compute gradient and do SGD step
            loss_total = loss + args.tri_lambda * loss_t + 0.1 * loss_kld +loss_kd_1 # +0*loss1
            # loss_total.requires_grad_(True)
            if 'cuda' in device:
                with amp.scale_loss(loss_total, optimizer) as loss_total:

                    loss_total.backward()
            else:
                loss_total.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()




        else:
            model2.train()

            output_t,feat_t,fmargin_t = model2(input_all, tag_all, return_feat=True)
            loss_kd_1 = loss_kd1(output, output_t)#最后一个先none
            loss_kd_2 = loss_kd2(output_t, output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target_all, topk=(1, 5))
            losses.update(loss.item(), input_all.size(0))
            losses_t.update(loss_t.item(), input_all.size(0))
            losses_t.update(loss_t.item(), input_all.size(0))
            avg_s_ap.update(s_ap.mean().item(), input_all.size(0))
            avg_s_an.update(s_an.mean().item(), input_all.size(0))
            top1.update(acc1[0], input_all.size(0))
            top5.update(acc5[0], input_all.size(0))

            #kld
            losses_kld.update(loss_kld.item(), input_all.size(0))
            losses_kd_1.update(loss_kd_1.item(), input_all.size(0))
            losses_kd_2.update(loss_kd_2.item(), input_all.size(0))
            # compute gradient and do SGD step
            loss_total = loss + args.tri_lambda*loss_t+0.1*loss_kld+loss_kd_1+loss_kd_2#+0*loss1
            #loss_total.requires_grad_(True)
            if 'cuda' in device:
                with amp.scale_loss(loss_total, optimizer) as loss_total:

                    loss_total.backward()
            else:
                loss_total.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()






        if i % args.print_freq == 0 or i == len(train_loader)-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_t.val:.3f} kld {loss_kld.val:.3f} kd {loss_kd1.val:.3f}'
                  '({loss.avg:.3f} {loss_t.avg:.3f} kld {loss_kld.avg:.3f} kd {loss_kd1.avg:.3f} \t'
                    'sp:{s_ap.avg:.3f} sn:{s_an.avg:.3f}) '
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, loss_t=losses_t,loss_kld=losses_kld,loss_kd1=losses_kd_1,
                s_ap=avg_s_ap, s_an=avg_s_an,

                top1=top1))
            #print('margin',s_mix)
            #print('circle {emd_loss.avg:.3f} '.format(emd_loss=losses1))
            #print('margin{}'.format(margin))

        #长度显示
        if epoch == 0 and i==len(train_loader)-200:
            plt.hist(s_an.cpu().detach().numpy().reshape(-1), bins=40,range=(0.7,1.5))
            plt.savefig('../img/anpic-{}-{}-{}.png'.format(epoch,args.dataset,args.loss))
            plt.close()
            plt.hist(s_ap.cpu().detach().numpy().reshape(-1), bins=40,color='orange',range=(0.7,1.5))
            plt.savefig('../img/appic-{}-{}-{}.png'.format(epoch,args.dataset,args.loss))
            print('saved')
        '''
        if epoch % 10 ==0 and i==len(train_loader)-200:
            print('margin', s_mix)
        '''
    if fix_margin=="unfix":
        fmargin =s_mix
        return fmargin

def validate(val_loader, model, args):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model_t.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.type(torch.LongTensor).view(-1,)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = model(input, torch.zeros(input.size()[0],1).to(device))

        # measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(val_loader)-1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time,
                top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg



def distance_view(target_all,tag_all,dist_mat):
    # print('tt',tag_all)
    same_poslist = []
    same_neglist = []
    notsame_poslist = []
    notsame_neglist = []
    is_pos = target_all[:, None].eq(target_all)

    is_same_modal = tag_all[:, None].eq(tag_all)
    dist_mat = dist_mat.cpu().detach()
    size = len(target_all)

    for i in range(size):
        for j in range(size):
            if is_pos[i][j]:
                same_poslist.append(dist_mat[i][j])

                '''    
                if is_same_modal[i][j]:
                    same_poslist.append(dist_mat[i][j])
                else:
                    notsame_poslist.append(dist_mat[i][j])
                '''

            else:
                same_neglist.append(dist_mat[i][j])
                '''
                if is_same_modal[i][j]:
                    same_neglist.append(dist_mat[i][j])
                else:
                    notsame_neglist.append(dist_mat[i][j])
                '''

    plt.scatter(same_poslist, same_neglist[::15], s=10, c='b')
    plt.xlim(0.1, 1.6)
    plt.ylim(0.1, 1.6)
    # plt.scatter(notsame_poslist, notsame_neglist[::15],s=10, c='g')
    plt.show()
    # print('1',sum(poslist)/len(poslist))
    # print('2',sum(neglist)/len(neglist))






def distance_view2(target_all, tag_all, dist_mat):
    # print('tt',tag_all)
    same_poslist = []
    same_neglist = []
    notsame_poslist = []
    notsame_neglist = []
    notmodal_neglist = []
    is_pos = target_all[:, None].eq(target_all)

    is_same_modal = tag_all[:, None].eq(tag_all)
    dist_mat = dist_mat.cpu().detach()
    size = len(target_all)
    # print(is_same_modal)
    for i in range(size):
        for j in range(size):
            if is_pos[i][j]:
                if i != j:

                    if is_same_modal[i][j] and len(same_poslist)==len(notsame_poslist):
                        same_poslist.append(dist_mat[i][j])
                    elif len(same_poslist)==(len(notsame_poslist)+1):
                        notsame_poslist.append(dist_mat[i][j])
                    else:
                        pass


            else:
                if is_same_modal[i][j] and len(same_neglist)==len(notsame_neglist):

                    same_neglist.append(dist_mat[i][j])
                elif len(same_neglist) == (len(notsame_neglist) + 1):
                    notsame_neglist.append((dist_mat[i][j]))
                else:
                    pass
                '''
                if is_same_modal[i][j]:
                    same_neglist.append(dist_mat[i][j])
                else:
                    notsame_neglist.append(dist_mat[i][j])
                '''

    '''#128*128,其中pos 1024个，同模不同模非各自512个（包括对角线全pos）
    test_same_poslist = []
    is_neg = target_all[:, None].ne(target_all)
    same_pos = is_neg&is_same_modal
    num=0

    for i in range(size):
        for j in range(size):
            if same_pos[i][j]:
                num+=1
    print('num',num)
    tri_num = 0
    for i in range(size):
        for j in range(i,size):
            if same_pos[i][j]:
                tri_num= tri_num+1
    print('trinum',tri_num)


    print('pos',len(same_poslist))
    print('neg1',len(same_neglist))
    print('neg2',len(notmodal_neglist))
    '''
    print('aaaa',len(same_poslist),len(notsame_poslist))

    l1=pd.Series(same_poslist,dtype=np.float64)
    l2=pd.Series(notsame_poslist,dtype=np.float64)
    corr = l1.corr(l2)
    print('corr',corr)
    plt.scatter(same_poslist[:240], notsame_poslist, s=2, c='b')
    plt.xlim(0.1, 1.6)
    plt.ylim(0.1, 1.6)
    # plt.scatter(notsame_poslist, notsame_neglist[::15],s=10, c='g')
    plt.show()


    l1=pd.Series(same_neglist,dtype=np.float64)
    l2=pd.Series(notsame_neglist,dtype=np.float64)

    corr = l1.corr(l2)
    print('corr2',corr)


    plt.scatter(same_neglist, notsame_neglist, s=2, c='g')
    plt.xlim(0.1, 1.6)
    plt.ylim(0.1, 1.6)
    # plt.scatter(notsame_poslist, notsame_neglist[::15],s=10, c='g')
    plt.show()




def distance_view3(target_all,tag_all,dist_mat):
    # print('tt',tag_all)
    same_poslist = []
    same_neglist = []
    notsame_poslist = []
    notsame_neglist = []
    notmodal_neglist=[]
    is_pos = target_all[:, None].eq(target_all)

    is_same_modal = tag_all[:, None].eq(tag_all)
    dist_mat = dist_mat.cpu().detach()
    size = len(target_all)
    #print(is_same_modal)
    for i in range(size):
        for j in range(size):
            if is_pos[i][j]:
                if i!=j:

                    if is_same_modal[i][j]:
                        same_poslist.append(dist_mat[i][j])
                    else:
                        notsame_poslist.append(dist_mat[i][j])


            else:
                if is_same_modal[i][j]:

                    same_neglist.append(dist_mat[i][j])
                else:
                    notmodal_neglist.append((dist_mat[i][j]))
                '''
                if is_same_modal[i][j]:
                    same_neglist.append(dist_mat[i][j])
                else:
                    notsame_neglist.append(dist_mat[i][j])
                '''


    '''#128*128,其中pos 1024个，同模不同模非各自512个（包括对角线全pos）
    test_same_poslist = []
    is_neg = target_all[:, None].ne(target_all)
    same_pos = is_neg&is_same_modal
    num=0

    for i in range(size):
        for j in range(size):
            if same_pos[i][j]:
                num+=1
    print('num',num)
    tri_num = 0
    for i in range(size):
        for j in range(i,size):
            if same_pos[i][j]:
                tri_num= tri_num+1
    print('trinum',tri_num)
    

    print('pos',len(same_poslist))
    print('neg1',len(same_neglist))
    print('neg2',len(notmodal_neglist))
    '''

    fig=plt.figure()
    ax = Axes3D(fig)

    x=np.array(same_neglist[1::20])
    y=np.array(notmodal_neglist[1::20])#15
    z=same_poslist

    ax.scatter(x, y, z, s=5, c='g')

    x=np.array(same_neglist[::20])
    y=np.array(notmodal_neglist[::20])
    z=notsame_poslist[:384]
    ax.scatter(x,y,z, s=5, c='b')
    ax.set_xlim(0.6,1)
    ax.set_ylim(0.6, 1)
    ax.set_zlim(0.4, 0.9)
    #ax.set_xlabel('$d_n1$')
    #ax.set_ylabel('$d_n2$')
    #ax.set_xlabel('$d_p$')
    #ax.text(x=1, y=0,z=0,s= 'd', fontsize=12)
    #plt.xlim(0.1, 1.6)
    #plt.ylim(0.1, 1.6)
    '''
    r = 0.43
    x0, y0, z0 = 1, 1, 0
    phi, theta = np.linspace(0, np.pi/2, 20), np.linspace(0,  np.pi/2, 40)
    PHI, THETA = np.meshgrid(phi, theta)
    X = x0 + r * np.sin(PHI) * np.cos(THETA)
    Y = y0 + r * np.sin(PHI) * np.sin(THETA)
    Z = z0 + r * np.cos(PHI)
    ax.plot_surface(X, Y, Z)
    # plt.scatter(notsame_poslist, notsame_neglist[::15],s=10, c='g')
    '''

    plt.show()
    # print('1',sum(poslist)/len(poslist))
    # print('2',sum(neglist)/len(neglist))