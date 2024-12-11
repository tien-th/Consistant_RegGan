import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class ImageDataset(Dataset):
    def __init__(self, root,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))

        
    def __getitem__(self, index):
        B_path = self.files_B[index % len(self.files_B)]
        A_path = self.files_A[index % len(self.files_A)]
        
        A_image = np.load(A_path, allow_pickle=True)
        B_image = np.load(B_path, allow_pickle=True)

        A_image = A_image / float(2047)
        B1_image = B_image[0] / float(32767)
        B2_image = B_image[1] / float(32767)

        A_image = Image.fromarray(A_image[0])
        B1_image = Image.fromarray(B1_image)
        B2_image = Image.fromarray(B2_image)
        
        A_image = self.transform(A_image)
        B1_image = self.transform(B1_image)
        B2_image = self.transform(B2_image)
        
        B1_image = (B1_image - 0.5) * 2.
        B2_image = (B2_image - 0.5) * 2.

        # concat B1 and A
        A_image = torch.cat((A_image, B1_image), 0)
        return {'A': A_image, 'B': B2_image}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        B_path = self.files_B[index % len(self.files_B)]
        A_path = self.files_A[index % len(self.files_A)]
        
        A_image = np.load(A_path, allow_pickle=True)
        B_image = np.load(B_path, allow_pickle=True)

        A_image = A_image / float(2047)
        B1_image = B_image[0] / float(32767)
        B2_image = B_image[1] / float(32767)

        A_image = Image.fromarray(A_image)
        B1_image = Image.fromarray(B1_image)
        B2_image = Image.fromarray(B2_image)
        
        A_image = self.transform(A_image)
        B1_image = self.transform(B1_image)
        B2_image = self.transform(B2_image)
        
        B1_image = (B1_image - 0.5) * 2.
        B2_image = (B2_image - 0.5) * 2.

        # concat B1 and A
        A_image = torch.cat((A_image, B1_image), 0)
        return {'A': A_image, 'B': B2_image, 'base_name': os.path.basename(A_path)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



# print('test')
# ds = ImageDataset('/home/PET-CT/tiennh/autopet256/train')
# print(len(ds))
# from torch.utils.data import DataLoader

# dl = DataLoader(ds, batch_size=1, shuffle=False)
# for i, data in enumerate(dl):
#     print('main', i)
#     print(data['A'].shape, data['B'].shape)
#     print(data['A'].max(), data['A'].min(), data['B'].max(), data['B'].min())
#     break