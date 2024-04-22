import clip
import torch
import torch.nn as nn
from senet import cse_resnet50_hashing
from resnet import resnet50_hashing
from global_tag import DTag
#from embedding import EMBEDDINGS
#from no_hubness.hub_args import parse_args
from embedding import embed_nohub

global args

'''
from utils import  get_train_args
global args
    # args = parser.parse_args()
args = get_train_args()
batch_size=args.batch_size
'''

class HashingModel(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, cse_end=4):
        super(HashingModel, self).__init__()

        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch

        if self.modelName == 'resnet50':
            self.original_model = resnet50_hashing(self.hashing_dim, pretrained=pretrained)
        elif self.modelName == 'se_resnet50':
            self.original_model = cse_resnet50_hashing(self.hashing_dim, cse_end=0, pretrained=pretrained)
        elif self.modelName == 'cse_resnet50':
            self.original_model = cse_resnet50_hashing(self.hashing_dim, cse_end=cse_end, pretrained=pretrained)


        self.original_model.last_linear = nn.Sequential()
        self.linear1 = nn.Linear(in_features=hashing_dim, out_features=num_classes, bias=False)
        self.linear2 = nn.Linear(in_features=hashing_dim,out_features=256,bias=False)
        self.linear3 = nn.Linear(in_features=256, out_features=128, bias=False)
        self.linear4 = nn.Linear(in_features=128, out_features=128, bias=False)
        self.linear5 = nn.Linear(in_features=128, out_features=3, bias=False)


    def forward(self, x, y, return_feat=False):
        DTag.set_domain_tag(y.int().squeeze())
        feats = self.original_model.features(x, y)
        feats = self.original_model.hashing(feats)
        #print(feats.size())
        out = self.linear1(feats)
        margin= torch.relu(self.linear2(feats))
        margin = torch.relu(self.linear3(margin))
        margin = torch.relu(self.linear4(margin))
        margin = self.linear5(margin)


        margin= torch.norm(margin,p=1,dim=0)
        #print(margin)
        '''
        min,_=torch.min(margin,dim=0)
        #print(min)
        max,_=torch.max(margin,dim=0)
        margin=(margin-min)/(max-min)
        margin = torch.mean(margin,dim=0)

        
        margin = margin.repeat(96,1)
        #print(margin)
        print(margin.size())
        #print(margin)
        #print('1',margin.size())
        margin = margin.clamp(min=1e-12).reshape(-1,1).squeeze()
        '''

        #margin =torch.nn.init.uniform_(margin, a=0.0, b=0.8)或许该正态
        #print(margin.size())
        if return_feat:
            return out, feats,margin
        else:
            return out



class CLIP(nn.Module):
    def __init__(self,hashing_dim=64,dropout_p=0.2):
        super(CLIP, self).__init__()
        self.model, preprocess = clip.load('ViT-B/32')
        #self.second_last_linear = nn.Linear(512, hashing_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

    def features(self, x, y, return_pyramid=False):
        pyramid = []
        DTag.set_domain_tag(y.int().squeeze())
        x = self.model.encode_image(x)
        if return_pyramid:
            x = x, pyramid

        return x
    def hashing(self, x):
        '''
        x = self.avgpool(x)
        if self.dropout is not None:  #resnet没有这个drop cse有
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.second_last_linear(x)
        先不hash
        '''
        if self.dropout is not None:  #resnet没有这个drop cse有
            x = self.dropout(x)
        return x

    '''
    def embed_features(self,features, episode=0):

        embeddings,loss = embed_nohub(
            features=features,  episode=episode
        )
        return embeddings,loss
    '''

    def forward(self,x,y):
        x = self.features(x, y)
        x = self.hashing(x)
        return(x)


class CLIPModel(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, cse_end=4):
        super(CLIPModel, self).__init__()


        self.num_classes = num_classes

        '''
        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch

        if self.modelName == 'resnet50':
            self.original_model = resnet50_hashing(self.hashing_dim, pretrained=pretrained)
        elif self.modelName == 'se_resnet50':
            self.original_model = cse_resnet50_hashing(self.hashing_dim, cse_end=0, pretrained=pretrained)
        elif self.modelName == 'cse_resnet50':
            self.original_model = cse_resnet50_hashing(self.hashing_dim, cse_end=cse_end, pretrained=pretrained)
        '''
        self.original_model= CLIP()
        self.original_model = self.original_model.float()




        self.linear1 = nn.Linear(in_features=512, out_features=num_classes, bias=False)
        self.linear2 = nn.Linear(in_features=512, out_features=256, bias=False)
        #self.linear1 = nn.Linear(in_features=1024, out_features=num_classes, bias=False)
        #self.linear2 = nn.Linear(in_features=1024, out_features=256, bias=False)

        self.linear3 = nn.Linear(in_features=256, out_features=128, bias=False)
        self.linear4 = nn.Linear(in_features=128, out_features=128, bias=False)
        self.linear5 = nn.Linear(in_features=128, out_features=3, bias=False)

    def forward(self, x, y, return_feat=False):
        DTag.set_domain_tag(y.int().squeeze())
        #feats = self.clipmodel.encode_image(x)
        feats = self.original_model.features(x,y)
        feats = self.original_model.hashing(feats)

        #feats,hub_loss =self.original_model.embed_features(feats,)


        # print(feats.size())
        out = self.linear1(feats)
        #在这里对feat处理？
        margin = torch.relu(self.linear2(feats))
        margin = torch.relu(self.linear3(margin))
        margin = torch.relu(self.linear4(margin))
        margin = self.linear5(margin)


        margin = torch.norm(margin, p=2, dim=0)
        # print(margin)
        '''
        min,_=torch.min(margin,dim=0)
        #print(min)
        max,_=torch.max(margin,dim=0)
        margin=(margin-min)/(max-min)
        margin = torch.mean(margin,dim=0)

        
        margin = margin.repeat(96,1)
        #print(margin)
        print(margin.size())
        #print(margin)
        #print('1',margin.size())
        margin = margin.clamp(min=1e-12).reshape(-1,1).squeeze()
        '''

        # margin =torch.nn.init.uniform_(margin, a=0.0, b=0.8)或许该正态
        # print(margin.size())
        if return_feat:
            return out, feats, margin
        else:
            return out



