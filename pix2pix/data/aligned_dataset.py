import os
import zarr
import glob
import torch
import torchvision
from data.base_dataset import BaseDataset


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    Modified to work with the LES dataset
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = self._get_sample_paths()  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def _get_sample_paths(self):
        sample_paths = []
        input_paths = sorted(glob.glob(f"{self.dir_AB}A/*o_input.zarr"))
        for input_path in input_paths:
            target_path = input_path.replace("A", "B").replace("input.zarr", "target.zarr")
            if os.path.exists(target_path):
                sample_paths.append((input_path, target_path))
        return sorted(sample_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - input tensor
            B (tensor) - - output tensor
            A_paths (str) - - input zarr store path
            B_paths (str) - - target zarr store path
        """
        # read a image given a random integer index
        A_paths, B_paths = self.AB_paths[index]
        A_store = zarr.open(A_paths, mode='r')
        B_store = zarr.open(B_paths, mode='r')
        A_vars = ['APCP_surface', 'REFC', 'UGRD_850mb', 'VGRD_850mb', 'DPT_850mb', 'TMP_850mb', 'UGRD_925mb', 'VGRD_925mb', 'DPT_925mb', 'TMP_925mb', 'TMP_surface', 'DPT_2m', 'elev', 'landsea']

        A = torch.nn.functional.pad(torch.stack([torch.from_numpy(A_store[var][2, :, :]) for var in A_vars], dim=0), (3,3))  # Shape: (14, 600, 250) padded to (14, 600, 256)
        B = torch.nn.functional.pad(torch.from_numpy(B_store['QPE_01H'][0, :, :]).unsqueeze(0), (3,3))  # Shape: (1, 600, 250) padded to (1, 600, 256)

        # A_precip = torch.nn.functional.pad((torch.from_numpy(A_store['APCP_surface'][2:5, :, :]).sum(dim=0).unsqueeze(0)), (3,3))
        # A_other = torch.nn.functional.pad((torch.stack([torch.from_numpy(A_store[var][i, :, :]) for var in A_vars for i in [2, 4]], dim=0)), (3,3))
        # A = torch.cat([A_precip, A_other], dim=0)  # Shape: (11, 600, 250) padded to (11, 600, 256)
        # B = torch.nn.functional.pad(torch.from_numpy(B_store['QPE_01H'][0:3, :, :]).sum(dim=0).unsqueeze(0), (3, 3))  # Shape: (1, 600, 250) padded to (1, 600, 256)

        # apply transforms if specified
        # if not self.opt.preprocess == 'none':
        #     transform = torchvision.transforms.RandomCrop(self.opt.crop_size)
        #     A = transform(A)
        #     B = transform(B)

        A = A[:, 250:250+256, :]
        A = torch.rot90(A, k=1, dims=(1, 2))
        B = B[:, 250:250+256, :]
        B = torch.rot90(B, k=1, dims=(1, 2))

        if torch.isnan(A).any(): print(f"NAN in INPUT {A_paths}")
        if torch.isnan(B).any(): print(f"NAN in TARGET{B_paths}")

        return {'A': A, 'B': B, 'A_paths': A_paths, 'B_paths': B_paths}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
