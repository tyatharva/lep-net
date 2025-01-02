#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:46:42 2024

@author: atyagi
"""

from subprocess import call

lakes = ['superior', 'michigan', 'erie', 'ontario']

for lake in lakes:
    call([
        "python",
        "test.py",
        "--dataroot", "./LES_dataset",
        "--name", f"{lake}_10",
        "--model", "pix2pix",
        "--input_nc", "10",
        "--output_nc", "1",
        "--init_type", "xavier",
        "--dataset_mode", f"{lake}",
        "--num_threads", "16",
        "--num_test", "3",
        "--netG", "unetformer",
        "--phase", "exp",
    ])
