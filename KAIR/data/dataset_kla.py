import os
import numpy as np
import torch
import torch.utils.data as data


class DatasetKLA(data.Dataset):
    """
    KLA .npy paired dataset for grayscale restoration.

    Input:
        NoisyLR/*.npy : 128 x 128 grayscale noisy image

    Target:
        GT/*.npy      : 256 x 256 grayscale clean image

    The dataset files are matched using the filename.
    Example:
        NoisyLR/000002.npy <-> GT/000002.npy
    """

    def __init__(self, opt):
        super(DatasetKLA, self).__init__()

        self.opt = opt
        self.phase = opt.get('phase', 'train')

        # --------------------------------------------------
        # Dataset paths
        # --------------------------------------------------
        self.dataroot_L = opt['dataroot_L']
        self.dataroot_H = opt['dataroot_H']

        # --------------------------------------------------
        # Find .npy files
        # --------------------------------------------------
        self.paths_L = sorted([
            os.path.join(self.dataroot_L, f)
            for f in os.listdir(self.dataroot_L)
            if f.lower().endswith('.npy')
        ])

        self.paths_H = sorted([
            os.path.join(self.dataroot_H, f)
            for f in os.listdir(self.dataroot_H)
            if f.lower().endswith('.npy')
        ])

        if not self.paths_L:
            raise RuntimeError(
                f'No .npy files found in NoisyLR folder: {self.dataroot_L}'
            )

        if not self.paths_H:
            raise RuntimeError(
                f'No .npy files found in GT folder: {self.dataroot_H}'
            )

        # --------------------------------------------------
        # Match files by filename
        # --------------------------------------------------
        H_dict = {
            os.path.basename(path): path
            for path in self.paths_H
        }

        paired_L = []
        paired_H = []

        for L_path in self.paths_L:
            filename = os.path.basename(L_path)

            if filename in H_dict:
                paired_L.append(L_path)
                paired_H.append(H_dict[filename])

        self.paths_L = paired_L
        self.paths_H = paired_H

        if not self.paths_L:
            raise RuntimeError(
                'No matching NoisyLR/GT .npy pairs were found.'
            )

        print('==========================================')
        print('KLA Dataset')
        print('==========================================')
        print('Phase        :', self.phase)
        print('NoisyLR path :', self.dataroot_L)
        print('GT path      :', self.dataroot_H)
        print('Pairs        :', len(self.paths_L))
        print('Input        : 128 x 128 x 1')
        print('Target       : 256 x 256 x 1')
        print('==========================================')

    def __getitem__(self, index):

        # --------------------------------------------------
        # Load paired .npy files
        # --------------------------------------------------
        L_path = self.paths_L[index]
        H_path = self.paths_H[index]

        img_L = np.load(L_path).astype(np.float32)
        img_H = np.load(H_path).astype(np.float32)

        # --------------------------------------------------
        # Check grayscale dimensions
        # --------------------------------------------------
        if img_L.ndim != 2:
            raise ValueError(
                f'Expected grayscale 2D NoisyLR array, '
                f'got shape {img_L.shape} in {L_path}'
            )

        if img_H.ndim != 2:
            raise ValueError(
                f'Expected grayscale 2D GT array, '
                f'got shape {img_H.shape} in {H_path}'
            )

        # --------------------------------------------------
        # Expected dimensions
        # --------------------------------------------------
        if img_L.shape != (128, 128):
            raise ValueError(
                f'Expected NoisyLR shape (128,128), '
                f'got {img_L.shape} in {L_path}'
            )

        if img_H.shape != (256, 256):
            raise ValueError(
                f'Expected GT shape (256,256), '
                f'got {img_H.shape} in {H_path}'
            )

        # --------------------------------------------------
        # Convert H,W -> 1,H,W
        # --------------------------------------------------
        img_L = torch.from_numpy(img_L).unsqueeze(0)
        img_H = torch.from_numpy(img_H).unsqueeze(0)

        return {
            'L': img_L,
            'H': img_H,
            'L_path': L_path,
            'H_path': H_path
        }

    def __len__(self):
        return len(self.paths_L)