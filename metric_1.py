import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
from tqdm import tqdm 


def compute_mae(image1, image2):
    return np.abs(image1 - image2).mean()

loss_fn_vgg = lpips.LPIPS(net='vgg')

gt_folder = '/home/PET-CT/splited_data_15k/test/B'
pre_folder = 'output/w_NC+R_55'

# MAX_PIXEL = 65535.0
MAX_PIXEL = 32767.0

ssim_scores = []
psnr_scores = []
mae_scores = []
lpips_scores = []

for filename in tqdm(os.listdir(gt_folder)):
    if filename.endswith(".npy"):
        # try:
        gt_path = os.path.join(gt_folder, filename)
        pre_path = os.path.join(pre_folder, filename)

        gt_img = np.load(gt_path, allow_pickle=True)
        pre_img = np.load(pre_path, allow_pickle=True)

            # print (gt_img.shape, pre_img.shape)
        # except:
        #     continue   

        pre_img = pre_img.mean(axis=-1) / MAX_PIXEL
        gt_img = np.squeeze(gt_img) * 0.5 + 0.5 
        
        pre_img = pre_img.astype(np.float32)
        gt_img = gt_img.astype(np.float32)

        # print (pre_img.shape, gt_img.shape, pre_img.min(), pre_img.max(), gt_img.min(), gt_img.max())
        
        ssim_score = ssim(pre_img, gt_img, data_range=1)
        psnr_score = psnr(pre_img, gt_img, data_range=1)
        mae = compute_mae(pre_img, gt_img)

        pre_img = torch.from_numpy(pre_img)
        pre_img = (pre_img - 0.5) * 2
        pre_img = pre_img.unsqueeze(0)
        pre_img = pre_img.expand(3, -1, -1)
        pre_img = pre_img.unsqueeze(0)
        
        gt_img = torch.from_numpy(gt_img)
        gt_img = (gt_img - 0.5) * 2
        gt_img = gt_img.unsqueeze(0)
        gt_img = gt_img.expand(3, -1, -1)
        gt_img = gt_img.unsqueeze(0)

        lpips_score = loss_fn_vgg(pre_img, gt_img) 
        lpips_score = lpips_score.detach().numpy()[0, 0, 0, 0]
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        mae_scores.append(mae)
        lpips_scores.append(lpips_score)

mean_ssim = np.mean(ssim_scores)
mean_psnr = np.mean(psnr_scores)
mean_mae = np.mean(mae_scores)
mean_lpips = np.mean(lpips_scores)

print("Mean SSIM: {}".format(mean_ssim))
print("Mean PSNR: {}".format(mean_psnr))
print("Mean MAE: {}".format(mean_mae * MAX_PIXEL))
print("Mean LPIPS: {}".format(mean_lpips))


# Mean SSIM: 0.9202779982399388
# Mean PSNR: 28.04688754811403
# Mean MAE: 643.1967894500121
# Mean LPIPS: 0.05809207260608673