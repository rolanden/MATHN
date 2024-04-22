import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import  torch.nn.functional as F
from test import evaluate
from utils import save_checkpoint, resume_from_checkpoint, \
    load_data, build_model_optm, get_train_args, fix_base_para
from loss import CrossMatchingTripletLoss, WeightedCrossMatchingTripletLoss
from sphereloss import CrossMatchingSphereLoss, WeightedCrossMatchingSphereLoss
from logger import Logger
from trainer import train, validate
import gc
from free_proxy import proxyfree
from circle_loss import CircleLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, input_logits, target_logits, mask=None,mask_pos=None):  # output_kd, output_t, tag_all, cid_mask_all * 0.3
        """
        :param input_logits: prediction logits
        :param target_logits: target logits
        :return: loss
        """
        log_likelihood = - F.log_softmax(input_logits, dim=1)

        if mask_pos is not None:
            target_logits = target_logits + mask_pos

        if mask is None:
            sample_num, class_num = target_logits.shape
            loss = torch.sum(torch.mul(log_likelihood, F.softmax(target_logits, dim=1))) / sample_num
        else:
            sample_num = torch.sum(mask)
            loss = torch.sum(torch.mul(torch.mul(log_likelihood, F.softmax(target_logits, dim=1)), mask)) / sample_num

        return loss



def main(passed_args=None):
    global args
    # args = parser.parse_args()
    args = get_train_args(passed_args)

    # generate exp name
    args.savedir = '{}/{}/{}/exp-{}-{}-m{}-TW{}-it{}-b{}-f{}/'.format(
        args.savedir, args.arch, args.dataset, args.remarks, args.loss, args.margin, args.tri_lambda,
        args.epoch_lenth, args.batch_size, args.num_hashing)
    args.resume_file = 'model_best.pth.tar'
    args.precision = True
    args.recompute = False
    args.pretrained = False
    if args.dataset == 'sketchy2':
        args.dataset = 'sketchy'
        args.zero_version = 'zeroshot2'


    sys.stdout = Logger(os.path.join(args.savedir, 'log.txt'))

    print(time.strftime('train-%Y-%m-%d-%H-%M-%S'))
    print(args)

    criterion_train = nn.CrossEntropyLoss()
    #free的那个
    #criterion_train = proxyfree()
    criterion_train_c = CircleLoss(scale=32,similarity='cos')

    criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                 normalize_feature=True,
                                                 mode='basic')

    '''改代理'''
    criterion_train_c= proxyfree()

    criterion_train_s = WeightedCrossMatchingSphereLoss(margin=args.margin,
                                                 normalize_feature=True,
                                                 mode='all')
    '''
    需要时候把下面train_t换成train_s
    
    '''


    if args.loss == 'ce':
        args.tri_lambda = 1
    elif args.loss == 'cross':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                     normalize_feature=True,
                                                     mode='basic')
    elif args.loss == 'within':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                     normalize_feature=True,
                                                     mode='within')
    elif args.loss == 'hybrid':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                     normalize_feature=True,
                                                     mode='partial')
    elif args.loss == 'all':
        criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
                                                     normalize_feature=True,
                                                     mode='all')
    elif args.loss == 'mathm':
        criterion_train_t = WeightedCrossMatchingTripletLoss(margin=args.margin,
                                                             normalize_feature=True,
                                                             mode='all')
    elif args.loss == 'easy':
        criterion_train_t = WeightedCrossMatchingTripletLoss(margin=args.margin,
                                                             normalize_feature=True,
                                                             mode='easy')
    # elif args.loss == 'ctri':
    #     args.ds_tri = False
    #     criterion_train_t = CrossMatchingTripletLoss(margin=args.margin,
    #                                                  normalize_feature=True,
    #                                                  mode=args.cross_mode)
    # elif args.loss == 'wctri':
    #     criterion_train_t = WeightedCrossMatchingTripletLoss(margin=args.margin,
    #                                                          normalize_feature=True,
    #                                                          mode=args.cross_mode)

    model, optimizer, scheduler = build_model_optm(args)

    optimizer.add_param_group({'params': list(criterion_train.parameters())})
    # scheduler.optimizer = optimizer

    resume_from_checkpoint(model, os.path.join(args.resume_dir, 'checkpoint.pth.tar'))


    #model2 teacher
    model2, optimizer2, scheduler2 = build_model_optm(args)

    optimizer2.add_param_group({'params': list(criterion_train.parameters())})
    # scheduler.optimizer = optimizer

    resume_from_checkpoint(model2, os.path.join(args.resume_dir, 'checkpoint.pth.tar'))

    criterion_train_kd = SoftCrossEntropy().cuda()
    criterion_train_kd_1 = SoftCrossEntropy().cuda()



    cudnn.benchmark = True




    mix_loader, val_loader,val_loader1 = load_data(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    best_acc1 = 0
    print('start training')
    for epoch in range(args.epochs):


        margin=0.3
        fix_margin=0.3
        params_not_to_train = ['linear1']
        '''
        if epoch>=60000 :#& epoch % 4!=0这里没用

            for name, param in model.named_parameters():
                # Set True only for params in the list 'params_to_train'model.named_parameters()
                #param.requires_grad = True if name in params_not_to_train else False
                if "linear1" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            #验证集拿来训练
            print('inner')
            for i in range(1):


                fix_margin=train(mix_loader, model, criterion_train, criterion_train_t,
                      optimizer, epoch, args,'unfix', margin)
                print(fix_margin)
        else:
            model.requires_grad_(True)
        '''
        model.requires_grad_(True)


        torch.cuda.empty_cache()

        if epoch < args.fixbase_epochs:    # fix pretrained para in first few epochs
            fix_base_para(model)
        else:
            model.requires_grad_(True)



        #检查更新情况
        '''
        for name, param in model.state_dict(keep_vars=True).items():
            print(name, param.requires_grad)
        '''

        print(epoch, *[param_group['lr'] for param_group in optimizer.param_groups])
        try:
            # model_t = None
            print('outer loop')
            #args.fix_margin="unfix"
            train(mix_loader, model, criterion_train, criterion_train_t,
                  optimizer, epoch, args,args.fix_margin,fix_margin,criterion_train_c,
                  model2,optimizer2,criterion_train_kd ,criterion_train_kd_1)

        except RuntimeError as e:
            raise e
            # print(e)
        acc1 = validate(val_loader, model, args)

        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.savedir, 'checkpoint.pth.tar'))
        if (epoch + 1) % args.eval_period == 0 or (epoch>7 and (epoch+1)%5==0):
            torch.cuda.empty_cache()
            del val_loader1
            del mix_loader
            gc.collect()
            evaluate(args, args.savedir, get_precision=True, model=model, recompute=True)
            mix_loader, val_loader, val_loader1 = load_data(args)

    args.itq = True
    # args.num_q_hashing = 64
    print('\n\n ----------------------  eval with itq -------------------------\n\n')
    evaluate(args, args.savedir, get_precision=True, model=model, recompute=False)


if __name__ == '__main__':

    main()

