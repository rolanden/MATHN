import os

import numpy
import numpy as np
from torch.utils.data import Dataset
from aug import random_transform, scatch_aug
import torchvision.transforms as transforms
import os.path as osp
from PIL import Image
import cv2

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(Dataset):
    def __init__(self, split='train',
                 root_dir='../dataset/Sketchy/',
                 version='sketch_tx_000000000000_ready', zero_version='zeroshot1',
                 cid_mask=False, transform=None, aug=False, shuffle=False, first_n_debug=9999999):

        self.root_dir = root_dir
        self.version = version
        self.split = split

        self.img_dir = self.root_dir

        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_test.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return

        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()

        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        #print('1112')
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        #print('lenth',len(self.labels))
        if shuffle:
            self.shuffle()

        self.file_ls = self.file_ls[:first_n_debug]
        self.labels = self.labels[:first_n_debug]

        self.transform = transform
        self.aug = aug
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.dataset = [os.path.join(self.img_dir, f) for f in self.file_ls]
        #print('type11',type(self.dataset))
        #print('sscas',len(self.dataset))

    def __len__(self):
        return len(self.labels)

    def get_path_and_label(self, idx):
        return os.path.join(self.img_dir, self.file_ls[idx]), self.labels[idx]

    def __getitem__(self, idx):
        #img = cv2.imread(os.path.join(self.img_dir, self.file_ls[idx]))[:,:,::-1]
        img = read_image(self.dataset[idx])
        #print('gg',type(img))

        '''
        #边缘图像
        img = numpy.array(img)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        scharrX = cv2.Scharr(img, -1, dx=1, dy=0)
        scharrY = cv2.Scharr(img, -1, dx=0, dy=1)
        scharrXABS = cv2.convertScaleAbs(scharrX)
        scharrYABS = cv2.convertScaleAbs(scharrY)
        img = cv2.addWeighted(scharrXABS, 0.5, scharrYABS, 0.5, 0)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        '''


        if self.transform is not None:
            img = self.transform(img)

        if self.aug == 'sketch':
            img = scatch_aug(img)
        elif self.aug == 'img':
            img = random_transform(img)
        else:
            pass



        img = self.normalize(img)

        label = self.labels[idx]
        #print(img.size())
        #print(label)
        return img, label

    def shuffle(self):
        s_idx = np.random.shuffle(np.arange(len(self.labels)))
        self.file_ls = self.file_ls[s_idx]
        self.labels = self.labels[s_idx]


class SketchyDataset(BaseDataset):

    def __init__(self, split='train',
                 root_dir='F:\zero-shot\zhj\MATHM-main\dataset\Sketchy/',
                 version='sketch_tx_000000000000_ready', zero_version='zeroshot1',
                 cid_mask=False, transform=None, aug=False, shuffle=False, first_n_debug=9999999):
        super().__init__(split, root_dir, version, zero_version, cid_mask, transform, aug, shuffle, first_n_debug)


class TUBerlinDataset(BaseDataset):
    def __init__(self, split='train',
                 root_dir='F:\zero-shot\zhj\MATHM-main\dataset\TUBerlin/',
                 version='png_ready', zero_version='zeroshot',
                 cid_mask=False, transform=None, aug=False, shuffle=False, first_n_debug=9999999):
        super().__init__(split, root_dir, version, zero_version, cid_mask, transform, aug, shuffle, first_n_debug)



class CUBDataset(BaseDataset):
    def __init__(self, split='train',
                 root_dir='../dataset/CUB_200_2011/',
                 version='images', zero_version='zeroshot',
                 cid_mask=False, transform=None, aug=False, shuffle=False, first_n_debug=9999999):
        super().__init__(split, root_dir, version, zero_version, cid_mask, transform, aug, shuffle, first_n_debug)

