#!/usr/bin/python3

import argparse
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from trainer import CycTrainerConsistency,P2p_Trainer,Nice_Trainer
import yaml


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGanConsistance.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'CycleGan':
        trainer = CycTrainerConsistency.Cyc_Trainer(config)

    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)
    # DATA_PATH = '/workdir/radish/PET-CT/data_gen/3D_LABELED_CT_PET_V2'
    txt_list_patients = '3d_dacthai/metric.txt'
    # read files 
    with open(txt_list_patients) as f:
        lines = f.readlines()
        patient_list = [x.strip() for x in lines]
    trainer._3D_inference(patient_list, '/workdir/radish/PET-CT/3D_reggan/metric')
    
    



###################################
if __name__ == '__main__':
    main()