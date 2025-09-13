import os
import cv2
import h5py
import torch
import random
import numpy as np
import torch.utils as u

def img2patch(path, win, stride, A):
    h, w, c = cv2.imread(os.path.join(path, '1_1.jpg')).shape
    total_num = (((h-win)//stride[0])+1) * (((w-win)//stride[1])+1)
    patches = np.zeros((total_num, c, win*A, win*A), dtype=np.float32)

    for i in range(A):
        for j in range(A):
            img = cv2.imread(os.path.join(path, str(i+1)+'_'+str(j+1)+'.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)

            for k in range(total_num):
                col = k % (((w-win)//stride[1])+1)
                row = k // (((w-win)//stride[1])+1)
                patches[k, :, i*win:(i+1)*win, j*win:(j+1)*win] \
                    = img[:, row*stride[0]:row*stride[0]+win, col*stride[1]:col*stride[1]+win]

    return patches


def crop_data(data_path, win, stride, A):

    print("Start cropping data")
    save_target_path = os.path.join(data_path, 'target.h5')
    save_input_path = os.path.join(data_path, 'input.h5')
    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    num = 0

    for scene in os.listdir(data_path):
        if scene.endswith('.h5'):
            continue
        scene_path = os.path.join(data_path, scene)

        target_path = os.path.join(scene_path, 'rain-free')
        target_patches = img2patch(target_path, win, stride, A)

        counter = target_patches.shape[0]

        for rainy in range(3):
            input_path = os.path.join(scene_path, 'rainy-%d' % (rainy+1))
            input_patches = img2patch(input_path, win, stride, A)

            for m in range(counter):
                input_data = input_patches[m, :, :, :]
                target_data = target_patches[m, :, :, :]
                input_h5f.create_dataset(str(num), data=input_data)
                target_h5f.create_dataset(str(num), data=target_data)
                num += 1
            print("Scene %s rainy-%d has been processed" % (scene, rainy+1))
    target_h5f.close()
    input_h5f.close()
    print("Finish cropping data,Total data : %d\n" % num)

class Dataset(u.data.Dataset):
    def __init__(self, mode, data_path='.'):
        super(Dataset, self).__init__()

        self.data_path = data_path
        self.mode = mode

        target_path = os.path.join(self.data_path, 'target.h5')
        input_path = os.path.join(self.data_path, 'input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        if self.mode == 'Train':
            random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'target.h5')
        input_path = os.path.join(self.data_path, 'input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)