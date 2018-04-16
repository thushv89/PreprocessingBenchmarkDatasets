import numpy as np
from PIL import Image
import os
import h5py

def test_by_saving_images_from_hdf5(hdf5file):

    dataset_file = h5py.File(hdf5file, "r")
    train_dataset = dataset_file['/train/images']

    if not os.path.exists('testing_trainset_images_of_svhn'):
        os.mkdir('testing_trainset_images_of_svhn')

    for f_i in range(5):
        rand_id = np.random.randint(10000)
        np_img = train_dataset[rand_id,:,:,:]
        #np_img = (np_img).astype(np.uint8)
        np_img -= np.min(np_img)
        np_img /= np.max(np_img)
        np_img = (np_img*255.0).astype(np.uint8)
        print(np_img)
        print(np.min(np_img))
        print(np.max(np_img))
        img = Image.fromarray(np_img,'RGB')
        img.save(os.path.join('testing_trainset_images_of_svhn','svhn_test_%d.jpg'%f_i))

if __name__=='__main__':
    test_by_saving_images_from_hdf5(os.path.join('data', 'svhn_10_dataset.hdf5'))