import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys


def gen_split(root_dir, splits, stack_size):
    DatasetX = []   # flow_x
    DatasetY = []   # flow_y
    DatasetF = []   # rgb
    Labels = []     # label
    NumFrames = []  # number of frames -> from flowDataset

    # root_dir  =  #drive/.../GTEA61/[processed_frames|flow_x_processed|flow_y_processed]/
    root_dir = os.path.join(root_dir, 'flow_x_processed')
    for split in splits: #
        dir1 = os.path.join(root_dir, split)  # root_dir/S#/
        class_id = 0
        for target in sorted(os.listdir(dir1)):
            if not target.startswith('.'):
                dir2 = os.path.join(dir1, target)  # root_dir/S#/target/
                insts = sorted(os.listdir(dir2))
                if insts:
                    for inst in insts:
                        if not inst.startswith('.'):
                            inst_dir = os.path.join(dir2, inst)  # root_dir/S#/target/#
                            num_frames = len(glob.glob1(inst_dir, '*[0-9].png'))
                            # TODO: check for elements with missing frames, note that the code is taking as number of
                            #       frames of the x_flow (check if this number coincides with rgb and flow_y frames)
                            if num_frames >= stack_size:
                                DatasetX.append(inst_dir)
                                DatasetY.append(inst_dir.replace('flow_x_processed',
                                                                 'flow_y_processed'))
                                DatasetF.append(inst_dir.replace('flow_x_processed',
                                                                 'processed_frames2'))  # local
                                                                #'processed_frames'))  # Colab
                                Labels.append(class_id)
                                NumFrames.append(num_frames)
                class_id += 1
    return DatasetX, DatasetY, DatasetF, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, splits,
                 spatial_transform=None,
                 sequence=False,
                 stackSize=5,
                 train=True,
                 numSeg=5,
                 fmt='.png',
                 phase='train',
                 seqLen = 25,
                 uniform_sampling=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imagesX, self.imagesY, self.imagesF, self.labels, self.numFrames = gen_split(
            root_dir, splits, stackSize)
        self.spatial_transform = spatial_transform
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = stackSize
        self.fmt = fmt
        self.phase = phase
        self.seqLen = seqLen
        self.uniform_sampling = uniform_sampling

    def __len__(self):
        return len(self.imagesX)

    def __getitem__(self, idx):
        vid_nameX = self.imagesX[idx]
        vid_nameY = self.imagesY[idx]
        vid_nameF = self.imagesF[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform.randomize_parameters()

        if numFrame <= self.stackSize:
            startFrame = 1
        else:
            if self.phase == 'train':
                startFrame = random.randint(1, numFrame - self.stackSize)
            else:
                startFrame = np.ceil((numFrame - self.stackSize)/2)


        # Collect the rgb and of frames
        inpSeqF = []
        inpSeq = []
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):

            # rgb
            fl_name = vid_nameF + '/rgb/rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqF.append(self.spatial_transform(img.convert('RGB')))

            # warp optical flow x
            fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))

            # warp optical flow y
            fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))

        inpSeqF = torch.stack(inpSeqF, 0)
        inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)

        return inpSeqSegs, inpSeqF, label #, vid_nameF#, fl_name
