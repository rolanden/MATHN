import random
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.transforms.functional as F
import argparse
import pickle
import os
import time
from random import randint
from PIL import Image
import torchvision
from render_sketch_chairv2 import redraw_Quick2RGB
from collections import defaultdict

# def get_ransform(opt):
#     transform_list = []
#     if opt.Train:
#         transform_list.extend([transforms.Resize(320), transforms.CenterCrop(299)])
#     else:
#         transform_list.extend([transforms.Resize(299)])
#     transform_list.extend(
#         [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     return transforms.Compose(transform_list)
def get_ransform(opt):
    transform_list = []
    if opt.Train:
        # n_rotate = random.random()
        # if n_rotate > 0.5:
        transform_list.extend([
        transforms.RandomRotation((-10,10), resample=2)])
            # n_crop = random.random()
            # if n_crop > 0.5:
            #     transform_list.extend([transforms.Resize(32), transforms.CenterCrop(28)])
            # elif n_crop > 0.3:
            #     transform_list.extend([transforms.Resize(64), transforms.CenterCrop(28)])
            # else:
            #     transform_list.extend([transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))])
        # else:
        transform_list.extend([transforms.Resize(32), transforms.CenterCrop(28)])
    else:
        transform_list.extend([transforms.Resize(224)]) #原来是28
    transform_list.extend(
        # [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return transforms.Compose(transform_list)

class CreateDataset_Sketchy(data.Dataset):
    def __init__(self, opt,mode,pic, on_Fly=False):
        with open(os.path.join(opt.roor_dir,opt.coordinate), 'rb') as fp:
            self.Coordinate = pickle.load(fp)
        #print(self.Coordinate)
        self.Skecth_Train_List = [x for x in self.Coordinate if 'train' in x]
        self.Skecth_Test_List = [x for x in self.Coordinate if 'test' in x]
        self.opt = opt
        self.transform = get_ransform(opt)
        self.on_Fly = on_Fly
        self.mode=mode
        self.pic=pic
        self.labels=[]
        #按情况把这里的labels改对
        '''
        all_photo_path = '../dataset/ShoeV2/photo'
        file_list = os.listdir(all_photo_path)
        all_label = [s.split('/')[-1].split('.')[0] for s in file_list]
        '''

        if self.mode == 'Train':
            if self.pic == 'sketchy':
                skt_label = [s.split('/')[-1].split('_')[0] for s in self.Skecth_Train_List]
                new_lst = []
                for i in skt_label:
                    new_lst.append(skt_label.index(i))
                    # print(new_lst)
                num = 0
                list1 = []
                new = 0
                for i in new_lst:
                    if i == new:
                        list1.append(num)
                    elif i != new:
                        num += 1
                        list1.append(num)
                    new = i
                self.labels = list1  # 草图标签
                sketch1_path = '../dataset/ShoeV2/trainA/'

                #sketch1_path = sketch1_path.join('/')
                file_list = os.listdir(sketch1_path)

                sketch_label=sketch1_path.join(file_list)
                #print(sketch_label)
                self.dataset = sketch_label

            elif self.pic == 'photo':
                '''
                photo1_path = '../dataset/ShoeV2/trainB'
                file_list = os.listdir(photo1_path)
                photo_label = [s.split('/')[-1].split('.')[0] for s in file_list]
            
                photo_train_list = []
                p = 0
                for m in range(p, 2000):

                    for i in range(0, 1800):
                        # print(photo_label[i])
                        if photo_label[i] == all_label[m]:
                            photo_train_list.append(m)
                            p += 1
                            break
                        elif i == 1800:
                            print('cnm', all_label[m])
                self.labels = photo_train_list
                '''
                self.labels = list(range(1800))


            else:
                print('e')
        elif self.mode== 'Test':
            if self.pic == 'sketchy':
                skt_label = [s.split('/')[-1].split('_')[0] for s in self.Skecth_Test_List]
                new_lst = []
                for i in skt_label:
                    new_lst.append(skt_label.index(i))
                    # print(new_lst)
                num = 0
                list1 = []
                new = 0
                for i in new_lst:
                    if i == new:
                        list1.append(num)
                    elif i != new:
                        num += 1
                        list1.append(num)
                    new = i
                self.labels = list1  # 草图标签

                sketch1_path = '../dataset/ShoeV2/testA/'

                #sketch1_path = sketch1_path.join('/')
                file_list = os.listdir(sketch1_path)

                sketch_label=sketch1_path.join(file_list)
                #print(sketch_label)
                self.dataset = sketch_label


            elif self.pic == 'photo':
                self.labels = list(range(200))
                sketch1_path = '../dataset/ShoeV2/testB/'

                #sketch1_path = sketch1_path.join('/')
                file_list = os.listdir(sketch1_path)

                sketch_label=sketch1_path.join(file_list)
                #print(sketch_label)
                self.dataset = sketch_label
                '''
                sketch1_path = '../dataset/ShoeV2/testA'

                file_list = os.listdir(sketch1_path)
                sketch_label = [s.split('/')[-1].split('_')[0] for s in file_list]
                # print(sketch_label)
                sketch_train_list = []
                p = 0
                for m in range(0, 2000):

                    for i in range(p, 666):
                        # print(photo_label[i])
                        if sketch_label[i] == all_label[m]:
                            sketch_train_list.append(m)

                self.labels = sketch_train_list
                '''
            else:
                print('e')
            #self.labels = list1

    def __getitem__(self, item):

        if self.mode == 'Train':

            sketch_path = self.Skecth_Train_List[item]




                #1对2000张图按文件名排序1-1999
                #2对训练测试分别用排序替代原位置

            all_photo_path = '../dataset/ShoeV2/trainB'
            file_list = os.listdir(all_photo_path)
            all_label = [s.split('/')[-1].split('.')[0] for s in file_list]

            '''
            index=[]
            for i  in all_label:
                index.append((i,(all_label.index(i))))
                #print(len(index))

                #训练照片
            photo1_path = '../dataset/ShoeV2/trainB'

            file_list = os.listdir(photo1_path)
            photo_label = [s.split('/')[-1].split('.')[0] for s in file_list]

            photo_train_list =[]
            p=0
            for m in range(p,1999):

                for i in range(0,1800):
                    #print(photo_label[i])
                    if photo_label[i]== all_label[m]:
                        photo_train_list.append(m)
                        p+=1
                        break
                    elif i == 1800:
                        print('cnm',all_label[m])

                #print('训练照片',photo_train_list)   #想办法拍一下
                #print('chang',len(photo_train_list))
                #print(photo_label)
            
            
                #训练草图
            sketch1_path = '../dataset/ShoeV2/trainA'

            file_list = os.listdir(sketch1_path)
            sketch_label = [s.split('/')[-1].split('_')[0] for s in file_list]
                #print(sketch_label)
            sketch_train_list = []
            p = 0
            for m in range(0, 1999):

                for i in range(p, 5982):
                    # print(photo_label[i])
                    if sketch_label[i] == all_label[m]:
                        sketch_train_list.append(m)
            
                #训练草图
            sketch1_path = '../dataset/ShoeV2/trainA'

            file_list = os.listdir(sketch1_path)
            sketch_label = [s.split('/')[-1].split('_')[0] for s in file_list]
                #print(sketch_label)
            sketch_train_list = []
            p = 0

            for m in range(0, 1800):
                flag = 0
                for i in range(p, 5982):
                    # print(photo_label[i])

                    if sketch_label[i] == all_label[m]:

                        sketch_train_list.append(m)
                        flag =1
                    if sketch_label[i]!=all_label[m] and  flag==1:
                        p = i
                        break
                


                #print('训练caotu', sketch_train_list)  # 想办法拍一下
                #print('长', len(sketch_train_list))
                # print(photo_label)




            skt_label = [s.split('/')[-1].split('_')[0] for s in self.Skecth_Train_List]
            new_lst = []
            for i in skt_label:
                new_lst.append(skt_label.index(i))
                #print(new_lst)
            num = 0
            list1=[]
            new = 0
            for i in new_lst:
                if i==new:
                    list1.append(num)
                elif i!=new:
                    num +=1
                    list1.append(num)
                new=i
            #self.labels=list1  #草图标签
            
                #print(list1)
                #print(positive_img)
                #print(skt_label)
                #print(new_lst)
                #print(positive_sample)
                #print(len(skt_label))
                #print('a',self.Skecth_Train_List)
                #print('a', self.Skecth_Train_List[item])
                #print('b',new_lst[item])
                #print('c',positive_sample)
            '''
            #print('草图的路径',self.Skecth_Train_List[item])
            #print('草图的编号',sketch_train_list[item])     #草图标签
            #print('照片的编号',positive_sample)

                #找照片的编号，有一说一照片和草图编号是一样的，没必要找两次
                #似乎不用找，直接按1-1800排
            '''
            p = 0
            for m in range(p, 1800):
                        # print(photo_label[i])
                if positive_sample == all_label[m]:
                    label = m
                    #print('编号',m)
                    break
                elif i == 1800:
                    print('cnm', positive_sample)
            





            #print('1')
            sample = {'sketch_img': sketch_img, 'sketch_path': self.Skecth_Train_List[item],
                        'positive_img': positive_img, 'positive_path': positive_sample,
                        'negetive_img': negetive_img, 'negetive_path': negetive_sample,
                        'Sample_Len': Sample_len}
            '''

            if self.pic == 'photo':
                positive_sample = '_'.join(self.Skecth_Train_List[item].split('/')[-1].split('_')[:-1])
                positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')


                possible_list = list(range(len(self.Skecth_Train_List)))
                possible_list.remove(item)
                flag = True
                while (flag):
                    negetive_item = possible_list[randint(0, len(possible_list) - 1)]
                    negetive_sample = '_'.join(self.Skecth_Train_List[negetive_item].split('/')[-1].split('_')[:-1])
                    if (negetive_sample != positive_sample):
                        flag = False

                negetive_path = os.path.join(self.opt.roor_dir, 'photo', negetive_sample + '.png')

                vector_x = self.Coordinate[sketch_path]
                sketch_img, Sample_len = redraw_Quick2RGB(vector_x)
                # print(Sample_len)
                if self.on_Fly == False:
                    sketch_img = Image.fromarray(sketch_img[-1].astype('uint8'))
                else:
                    sketch_img = [Image.fromarray(sk_img.astype('uint8')) for sk_img in sketch_img]

                positive_img = Image.open(positive_path)
                negetive_img = Image.open(negetive_path)

                n_flip = random.random()
                if n_flip > 0.5:

                    if self.on_Fly == False:
                        sketch_img = F.hflip(sketch_img)
                    else:
                        sketch_img = [F.hflip(sk_img) for sk_img in sketch_img]

                    positive_img = F.hflip(positive_img)
                    negetive_img = F.hflip(negetive_img)

                if self.on_Fly == False:
                    sketch_img = self.transform(sketch_img)
                else:
                    sketch_img = [self.transform(sk_img) for sk_img in sketch_img]

                positive_img = self.transform(positive_img)
                negetive_img = self.transform(negetive_img)

                img=positive_img
                #self.labels=list1
                label = item
                #print('zhaop',img.size())


            if self.pic == 'sketchy':

                sketchy_path= '../dataset/ShoeV2/trainA'
                sketchy_name=   self.Skecth_Train_List[item].split('/')[-1]
                sketchy_path= os.path.join(sketchy_path,sketchy_name+'.png')
                #print(sketchy_path)


                p = 0
                for m in range(p, 1800):
                    # print(photo_label[i])
                    if sketchy_name.split('_')[0]== all_label[m]:
                        label = m
                        # print('编号',m)
                        break
                    elif m == 1800:
                        print('cnm', positive_sample)


                # Open an image file
                with Image.open(sketchy_path) as im:

                    # Convert image to RGB
                    sketch_img = im.convert('RGB')



                #sketch_img = Image.open(sketchy_path)


                n_flip = random.random()
                if n_flip > 0.5:


                    sketch_img = F.hflip(sketch_img)


                sketch_img = self.transform(sketch_img)


                img=sketch_img
                #self.labels=list1
                label=label
                #print('sk',img.size())

                #print(img)
                #if isinstance(img, list):
                 #   print(img)



        elif self.mode == 'Test':


            sketch_path = self.Skecth_Test_List[item]




            all_photo_path = '../dataset/ShoeV2/testB'
            file_list = os.listdir(all_photo_path)
            all_label = [s.split('/')[-1].split('.')[0] for s in file_list]










            '''
            sample = {'sketch_img': sketch_img, 'sketch_path': self.Skecth_Test_List[item],
                      'positive_img': positive_img,
                      'negetive_img': negetive_img, 'negetive_path': negetive_sample,
                      'positive_path': positive_sample, 'Sample_Len': Sample_len}
                      
            '''
        #print('hh', len(sketch_img))
            if self.pic == 'photo':

                positive_sample = '_'.join(self.Skecth_Test_List[item].split('/')[-1].split('_')[:-1])
                positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')
                possible_list = list(range(len(self.Skecth_Test_List)))
                possible_list.remove(item)
                flag = True
                while (flag):
                    negetive_item = possible_list[randint(0, len(possible_list) - 1)]
                    negetive_sample = '_'.join(self.Skecth_Train_List[negetive_item].split('/')[-1].split('_')[:-1])
                    if (negetive_sample != positive_sample):
                        flag = False

                negetive_path = os.path.join(self.opt.roor_dir, 'photo', negetive_sample + '.png')
                vector_x = self.Coordinate[sketch_path]
                sketch_img, Sample_len = redraw_Quick2RGB(vector_x)
                if self.on_Fly == False:
                    sketch_img = self.transform(Image.fromarray(sketch_img[-1].astype('uint8')))
                else:
                    sketch_img = [self.transform(Image.fromarray(sk_img.astype('uint8'))) for sk_img in sketch_img]

                positive_img = self.transform(Image.open(positive_path))
                negetive_img = self.transform(Image.open(negetive_path))

                img = positive_img
                label = item

            if self.pic == 'sketchy':

                sketchy_path= '../dataset/ShoeV2/testA'
                sketchy_name=   self.Skecth_Test_List[item].split('/')[-1]
                sketchy_path= os.path.join(sketchy_path,sketchy_name+'.png')
                #print(sketchy_path)


                for m in range(0, 201):
                    # print(photo_label[i])
                    if sketchy_name.split('_')[0]== all_label[m]:
                        label = m
                        # print('编号',m)
                        break
                    elif m == 200:
                        print('cnm', positive_sample)


                # Open an image file
                with Image.open(sketchy_path) as im:

                    # Convert image to RGB
                    sketch_img = im.convert('RGB')



                #sketch_img = Image.open(sketchy_path)


                n_flip = random.random()
                if n_flip > 0.5:


                    sketch_img = F.hflip(sketch_img)


                sketch_img = self.transform(sketch_img)

                img = sketch_img
                #label = label

        return img, label
        #print(label)
        #return sample


    def __len__(self):
        if self.mode == 'Train':
            if self.pic == 'photo':
                return 1800
            if self.pic == 'sketchy':
                return 5982
            #return len(self.Skecth_Train_List)
        elif self.mode == 'Test':
            if self.pic == 'photo':
                return 200
            if self.pic == 'sketchy':
                return 666
            #return len(self.Skecth_Test_List)

    def get_img_info(data_dir):
        data_info = list()


        return data_info



class ShoeDataset(CreateDataset_Sketchy):
    def __init__(self, args,mode,pic, on_Fly=True):
        super().__init__(args,mode,pic, on_Fly)



if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = 'ShoeV2_Coordinate'
    opt.roor_dir = '../dataset/ShoeV2'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.nThreads = 1
    opt.pic='photo'
    opt.batchsize = 16
    dataset_sketchy = CreateDataset_Sketchy(opt, on_Fly=True)

    dataloader_sketchy = data.DataLoader(dataset_sketchy, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                         num_workers=int(opt.nThreads))

    print('1')

    support = [2.0, 2.0, 3.0, 1.0, 1.0]
    for i in support:
        print(support.index(i))


    for batch_idx, samples in enumerate(dataloader_sketchy):


        print(batch_idx)



        #print(batch_idx, samples['sketch_path'])


    for i_batch, sanpled_batch in enumerate(dataloader_sketchy):
        t0 = time.time()
        if i_batch == 1:
            print(len(sanpled_batch['sketch_img'][0]))
        torchvision.utils.save_image(sanpled_batch['sketch_img'][-1], 'sketch_img.jpg', normalize=True)
        torchvision.utils.save_image(sanpled_batch['positive_img'], 'positive_img.jpg', normalize=True)
        # torchvision.utils.save_image(sanpled_batch['negetive_img'], 'negetive_img.jpg', normalize=True)
        # print(i_batch, sanpled_batch['class_label'], (time.time() - t0))
        # print(sanpled_batch['sketch_img'][-1][0])
        # for i_num in range(len(sanpled_batch['sketch_img'])):
        #    torchvision.utils.save_image(sanpled_batch['sketch_img'][i_num], str(i_num) + 'sketch_img.jpg',
        #                                 normalize=True)
'''