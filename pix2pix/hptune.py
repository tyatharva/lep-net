#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:28:10 2024

@author: atyagi
"""

from subprocess import call, DEVNULL
from tqdm import tqdm

gans = ['lsgan', 'wgangp']
schs = ['linear', 'cosine']
lrs = [0.0002, 0.0005, 0.001]
loss_fns = ['l1', 'l2', 'huber', 'wl2', 'ssiml2', 'hybrid']
l1s = [25, 100, 500, 1000]

# Calculate total iterations
total_iterations = len(gans) * len(schs) * len(lrs) * len(loss_fns) * len(l1s)

# Open the error log file
with open('err.txt', 'w') as err_file:
    # Wrap the entire loop with tqdm for overall progress
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for gan in gans:
            for sch in schs:
                for lr in lrs:
                    for loss_fn in loss_fns:
                        for l1 in l1s:
                            # Run the training script
                            call([
                                "python",
                                "train.py",
                                "--dataroot", "./LES_dataset",
                                "--name", f"{gan}_{sch}_{lr}_{loss_fn}_{l1}",
                                "--model", "pix2pix",
                                "--input_nc", "14",
                                "--output_nc", "1",
                                "--norm", "batch",
                                "--init_type", "xavier",
                                "--dataset_mode", "aligned",
                                "--num_threads", "16",
                                "--batch_size", "64",
                                "--n_epochs", "20",
                                "--n_epochs_decay", "5",
                                "--save_epoch_freq", "50",
                                "--netG", "unetformer",
                                "--lr_policy", f"{sch}",
                                "--lr", f"{lr}",
                                "--lambda_L1", f"{l1}",
                                "--loss_fn", f"{loss_fn}",
                                "--gan_mode", f"{gan}"
                            ], stdout=DEVNULL, stderr=err_file)
        
                            # Run the test script
                            call([
                                "python",
                                "test.py",
                                "--dataroot", "./LES_dataset",
                                "--name", f"{gan}_{sch}_{lr}_{loss_fn}_{l1}",
                                "--model", "pix2pix",
                                "--input_nc", "14",
                                "--output_nc", "1",
                                "--init_type", "xavier",
                                "--dataset_mode", "test",
                                "--num_threads", "16",
                                "--num_test", "64",
                                "--netG", "unetformer",
                            ], stdout=DEVNULL, stderr=err_file)
        
                            # Update the progress bar
                            pbar.update(1)