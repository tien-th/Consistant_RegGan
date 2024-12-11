#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import ImageDataset,ValDataset
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# use tensorboard 
from torch.utils.tensorboard import SummaryWriter

class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['input_RA']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_RA'],config['input_RA']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = Discriminator(config['input_RA']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        level = config['noise_level']  # set noise level
        

        self.dataloader = DataLoader(ImageDataset(config['dataroot']),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], drop_last=True)

        val_transforms = [ToTensor(),
                          Resize(size_tuple = (config['size'], config['size']))]
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot']),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'], drop_last=True)

 
       # Loss plot
        # self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))   
        #  use tensorboard
        self.logger = SummaryWriter(log_dir = config['save_root'] + '/log')
        os.makedirs(config['save_root'] + '/log', exist_ok=True)
        
    def train(self):
        ###### Training ######
        if not os.path.exists(self.config["save_root"]):
            os.makedirs(self.config["save_root"])
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            epoch_loss = 0
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))

                if self.config['bidirect']:   # C dir
                    if self.config['regist']:    #C + R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        
                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        
                        Trans = self.R_A(fake_B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        
                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss +SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()
                        
                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ################################### 
                    
                    else: #only  dir:  C
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda']*self.MSE_loss(pred_fake, self.target_real)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                        loss_Total.backward()
                        self.optimizer_G.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################     
                else:                  # s dir :NC
                    if self.config['regist']:    # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)
                        
                        Trans = self.R_A(fake_B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        
                        Trans1 = self.R_A(fake_B, real_A[:, 1, :, :].unsqueeze(1))
                        Trans_gt = self.R_A(real_B, real_A[:, 1, :, :].unsqueeze(1))

                        consistency_loss = self.config['Consistency_lambda'] * torch.nn.MSELoss()(Trans1, Trans_gt)

                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        ####smooth loss 
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        toal_loss = SM_loss+adv_loss+SR_loss + consistency_loss
                        toal_loss.backward()
                        epoch_loss += toal_loss.item()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)


                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        
                    # log to tensorboard
                    self.logger.add_scalar('SM_loss', SM_loss.item(), epoch * len(self.dataloader) + i)
                    self.logger.add_scalar('adv_loss', adv_loss.item(), epoch * len(self.dataloader) + i)
                    self.logger.add_scalar('SR_loss', SR_loss.item(), epoch * len(self.dataloader) + i)
                    self.logger.add_scalar('consistency_loss', consistency_loss.item(), epoch * len(self.dataloader) + i)
                    self.logger.add_scalar('toal_loss', toal_loss.item(), epoch * len(self.dataloader) + i)
                    print(f"Epoch {epoch}, SM_loss: {SM_loss.item()}, adv_loss: {adv_loss.item()}, SR_loss: {SR_loss.item()}, consistency_loss: {consistency_loss.item()}, toal_loss: {toal_loss.item()}")

            self.logger.add_scalar('epoch_loss', epoch_loss, epoch)
            print(f"Epoch {epoch}, epoch_loss: {epoch_loss}")
            # save for resume training
            checkpoint = {
                epoch: epoch,
                'netG_A2B': self.netG_A2B.state_dict(),
                'optimizer_G': self.optimizer_G.state_dict(),
                'netD_B': self.netD_B.state_dict(),
                'optimizer_D_B': self.optimizer_D_B.state_dict(),
                'R_A': self.R_A.state_dict(),
                'optimizer_R_A': self.optimizer_R_A.state_dict()
            }
            # save model
            torch.save(checkpoint, self.config['save_root'] + 'last_checkpoint.pth')

                                    
    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B_55.pth'))
        #self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        
        with torch.no_grad():

                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()  
                    for j in range(fake_B.shape[0]):
                        filename = batch['base_name'][j] 
                        self.save_result(fake_B[j],'output/w_NC+R_55' , filename)
                    #     break
                    # break

    def _3D_inference(self, patient_list, result_path):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B_198.pth'))

        def preprocess(ct_slice, x_d_minus_one): # ct_voxel: W * H (512 * 512)
            transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(256),
            transforms.ToTensor()
            ])
            ct = ct_slice / float(2047)
            A_image = Image.fromarray(ct)
            A_image = transform(A_image)

            x_d_minus_one = x_d_minus_one / float(32767)

            B1_image = Image.fromarray(x_d_minus_one)
            B1_image = transform(B1_image)
            B1_image = (B1_image - 0.5) * 2.
            A_image = torch.cat((A_image, B1_image), 0)
            return A_image
        
        def postprocess(fake_B, max_pixel = 32767):
            fake_B = fake_B.detach().cpu().numpy().squeeze()  
            image = fake_B
            image = (image * 0.5 + 0.5).clip(0, 1)
            image = (image * max_pixel).clip(0, max_pixel)
            return image
        # Duyệt qua từng thư mục bệnh nhân trong DATA_PATH
        for patient_folder in tqdm(patient_list):
            patient_path = patient_folder
            
            # Kiểm tra xem có phải là thư mục không
            if os.path.isdir(patient_path):
                # Tìm file ct.npy bên trong thư mục bệnh nhân
                ct_file_path = os.path.join(patient_path, 'ct.npy')
                pet_file_path = os.path.join(patient_path, 'pet.npy')
                
                # Kiểm tra tệp có tồn tại hay không
                if os.path.exists(ct_file_path):
                    ct_img = np.load(ct_file_path, allow_pickle=True)
                    x_d_minus_one = np.load(pet_file_path, allow_pickle=True)[0]
                    predicted_slices = [x_d_minus_one]

                    # Lặp qua các lát cắt để dự đoán
                    
                    for i in tqdm(range(1, ct_img.shape[0])):
                        ct_slice = ct_img[i]
                        A_image = preprocess(ct_slice, predicted_slices[-1])
                        # print(A_image.unsqueeze(0).shape)
                        
                        real_A = Variable(self.input_A.copy_(A_image))
                        fake_B = self.netG_A2B(real_A)
                        fake_B = postprocess(fake_B)
                        predicted_slices.append(fake_B)
                    # Chuyển danh sách các lát cắt đã dự đoán thành một khối 3D numpy array
                    predicted_volume = np.stack(predicted_slices, axis=0)

                    # Tạo thư mục kết quả riêng cho bệnh nhân nếu chưa tồn tại
                    patient_result_path = os.path.join(result_path, os.path.basename(patient_folder))
                    os.makedirs(patient_result_path, exist_ok=True)
                    # print(f"Saving result to {patient_result_path}")
                    # Lưu kết quả dự đoán vào thư mục của bệnh nhân
                    output_file_path = os.path.join(patient_result_path, 'predicted_volume_registered.npy')
                    np.save(output_file_path, predicted_volume)
                    print(f"Saved predicted volume to {output_file_path}")

    def save_result(self, fake, outdir, filename):
        os.makedirs(outdir, exist_ok=True)
        image = fake
        image = (image * 0.5 + 0.5).clip(0, 1)
        # image = (image * 65535).clip(0, 65535)
        image = (image * 32767) + 0.2
        image =  image.clip(0, 32767)
        # image = np.transpose(image, (1, 2, 0))
        image = image = np.expand_dims(image, axis=-1) 
        # print(os.path.join(outdir, filename))
        np.save(os.path.join(outdir, filename), image)

  
    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 

