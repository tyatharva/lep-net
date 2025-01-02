#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:46:42 2024

@author: atyagi
"""

from subprocess import call, DEVNULL
from tqdm import tqdm

lakes = ['superior', 'michigan', 'erie', 'ontario']
varnums = [2, 5, 10, 11, 14]
testnums = [120, 237, 262, 333]
total_iterations = len(lakes) * len(varnums)

with open('err.txt', 'w') as err_file:
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for lake in lakes:
            testnum = testnums[lakes.index(lake)]
            for varnum in varnums:
                # call([
                #     "python",
                #     "train.py",
                #     "--dataroot", "./LES_dataset",
                #     "--name", f"{lake}_{varnum}",
                #     "--model", "pix2pix",
                #     "--input_nc", f"{varnum}",
                #     "--output_nc", "1",
                #     "--norm", "batch",
                #     "--init_type", "xavier",
                #     "--dataset_mode", f"{lake}",
                #     "--num_threads", "16",
                #     "--batch_size", "64",
                #     "--n_epochs", "45",
                #     "--n_epochs_decay", "5",
                #     "--save_epoch_freq", "55",
                #     "--netG", "unetformer",
                #     "--lr_policy", "linear",
                #     "--lr", "0.001",
                #     "--lambda_L1", "500",
                #     "--loss_fn", "l1",
                #     "--gan_mode", "lsgan",
                #     "--phase", "train",
                # ], stdout=DEVNULL, stderr=err_file)

                call([
                    "python",
                    "test.py",
                    "--dataroot", "./LES_dataset",
                    "--name", f"{lake}_{varnum}",
                    "--model", "pix2pix",
                    "--input_nc", f"{varnum}",
                    "--output_nc", "1",
                    "--init_type", "xavier",
                    "--dataset_mode", f"{lake}",
                    "--num_threads", "16",
                    "--num_test", f"{testnum}",
                    "--netG", "unetformer",
                    "--phase", "test",
                ], stdout=DEVNULL, stderr=err_file)

                pbar.update(1)
