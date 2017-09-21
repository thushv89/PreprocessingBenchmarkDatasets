import pickle
import os
import numpy as np
import h5py
from PIL import Image
import multiprocessing as mp
from functools import partial
from sklearn.preprocessing import StandardScaler

def save_cifar10_as_hdf5(data_dir, original_image_w, resize_images_to, save_dir):

    print('\nProcessing Cifar-10 Data')

    hdf5_sep = '/'
    n_train = 50000
    n_test = 10000
    n_channels = 3

    dataset_filename = save_dir + os.sep + 'cifar_10_dataset.hdf5'

    filenames = [os.path.join(data_dir, 'cifar_10_data_batch_%d' % i) for i in range(1, 6)]

    print('Created Training File')
    hdf5_file = h5py.File(dataset_filename, "w")
    dataset_names = hdf5_file.keys()

    # =======================================================================
    # Saving Training Data
    train_dataset,train_labels = None,None
    for fname in filenames:
        print('Processing file: %s'%fname)
        with open(fname, 'rb') as f:
            save = pickle.load(f, encoding='latin1')
            dataset, labels = save['data'], save['labels']

            # Reshaping images from 1-D to 3-D and converting labels to ndarray
            dataset = dataset.reshape((-1, original_image_w, original_image_w, n_channels), order='F').astype(np.uint8)
            dataset = np.transpose(dataset, [0, 2, 1, 3])
            labels = np.reshape(labels,(-1,))
            if train_dataset is None:
                train_dataset = dataset
                train_labels = labels
            else:
                train_dataset = np.append(train_dataset, dataset, axis=0)
                train_labels = np.append(train_labels, labels, axis=0)

            print('\tTrain file size (images): %d' %dataset.shape[0])
            print('\tTrain file size (labels): %d' % labels.shape[0])


    # =====================================================================
    # Saving Training Data to HDF5
    # =====================================================================

    if not ('train' + hdf5_sep + 'images' in dataset_names and 'train' + hdf5_sep + 'labels' in dataset_names):
        hdf5_train_group = hdf5_file.create_group('train')
        hdf5images = hdf5_train_group.create_dataset('images', (n_train, resize_images_to,
                                                                resize_images_to, n_channels), dtype='f')
        hdf5labels = hdf5_train_group.create_dataset('labels', (n_train, 1), dtype='int32')

        train_dataset, train_labels = process_dataset_with_multiprocessing(train_dataset,train_labels, resize_images_to)

        # Normalize data
        train_dataset -= np.reshape(np.mean(train_dataset, axis=(1, 2, 3)), (-1, 1, 1, 1))
        train_dataset /= np.reshape(np.std(train_dataset, axis=(1, 2, 3)), (-1, 1, 1, 1))

        hdf5images[:, :, :, :] = train_dataset
        hdf5labels[:,0] = train_labels[:,0]

        del train_dataset,train_labels

    # ================================================================================================
    print('Currently available keys in the HDF5 File: ')
    print(hdf5_file.keys())

    # =====================================================================================
    # Saving Testing Data
    with open(os.path.join(data_dir, 'cifar_10_test_batch'), 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        test_dataset, test_labels = save['data'], save['labels']

        # Reshaping images from 1-D to 3-D and converting labels to ndarray
        test_dataset = test_dataset.reshape((-1, original_image_w, original_image_w, n_channels), order='F').astype(np.uint8)
        test_dataset = np.transpose(test_dataset, [0, 2, 1, 3])
        test_labels = np.asarray(test_labels).reshape(-1,1)
    # =====================================================================
    # Saving Testing Data to HDF5
    # =====================================================================
    if not ('test' + hdf5_sep + 'images' in dataset_names and 'test' + hdf5_sep + 'labels' in dataset_names):
        hdf5_train_group = hdf5_file.create_group('test')
        hdf5_test_images = hdf5_train_group.create_dataset('images', (n_test, resize_images_to,
                                                                resize_images_to, n_channels), dtype='f')
        hdf5_test_labels = hdf5_train_group.create_dataset('labels', (n_test, 1), dtype='int32')

        test_dataset, test_labels = process_dataset_with_multiprocessing(test_dataset, test_labels, resize_images_to)
        # Normalize data
        test_dataset -= np.reshape(np.mean(test_dataset, axis=(1, 2, 3)), (-1, 1, 1, 1))
        test_dataset /= np.reshape(np.std(test_dataset, axis=(1, 2, 3)), (-1, 1, 1, 1))

        hdf5_test_images[:, :, :, :] = test_dataset
        hdf5_test_labels[:, 0] = test_labels[:,0]


def process_dataset_with_multiprocessing(dataset_images, dataset_labels, resize_images_to):
    part_preprocess_images_func = partial(preprocess_images_by_one, resize_to=resize_images_to)

    # do not use all the CPUs if there are a lot only use half of them
    # if using all, leave one free
    cpu_count = mp.cpu_count() - 1 if mp.cpu_count() < 32 else mp.cpu_count() // 2
    pool = mp.Pool(cpu_count)

    print('Train dataset size before using multiprocessing: ', dataset_images.shape)
    list_dataset = np.vsplit(dataset_images, dataset_images.shape[0])
    list_labels = dataset_labels.tolist()
    print('Size of a single item after using vsplit operation: ', list_dataset[0].shape)
    preproc_data = pool.starmap(part_preprocess_images_func, zip(list_dataset,list_labels), chunksize=dataset_images.shape[0]//cpu_count)
    preproc_images, preproc_labels = zip(*preproc_data)
    print('Size of the returned list by multiprocessing map operation: ', len(preproc_data))
    dataset_images = np.stack(preproc_images, axis=0)
    dataset_labels = np.asarray(preproc_labels).reshape(-1,1)
    print('Size of the array after stacking the items in the list: ', dataset_images.shape)

    return dataset_images,dataset_labels


def save_cifar100_as_hdf5(data_dir, original_image_w, resize_images_to, save_dir):

    print('\nProcessing Cifar-100 Data')
    hdf5_sep = '/'
    n_train = 50000
    n_test = 10000
    n_channels = 3

    dataset_filename = save_dir + os.sep + 'cifar_100_dataset.hdf5'

    train_filename = data_dir + os.sep+'train'
    test_filename = data_dir + os.sep + 'test'

    hdf5_file = h5py.File(dataset_filename, "w")
    dataset_names = hdf5_file.keys()

    # =======================================================================
    # Saving Training Data

    with open(train_filename, 'rb') as f:
        save = pickle.load(f, encoding="latin1")
        train_dataset, train_labels = np.asarray(save['data']), np.asarray(save['fine_labels'])

        # Reshaping images from 1-D to 3-D and converting labels to ndarray
        train_dataset = train_dataset.reshape((-1, original_image_w, original_image_w, n_channels), order='F').astype(np.uint8)
        train_dataset = np.transpose(train_dataset, [0, 2, 1, 3])
        train_dataset -= np.mean(train_dataset, axis=(1, 2, 3))
        train_dataset /= np.std(train_dataset, axis=(1, 2, 3))
        train_labels = train_labels.reshape(-1,)

    # =====================================================================
    # Saving Training Data to HDF5
    # =====================================================================

    if not ('train' + hdf5_sep + 'images' in dataset_names and 'train' + hdf5_sep + 'labels' in dataset_names):
        hdf5_train_group = hdf5_file.create_group('train')
        hdf5images = hdf5_train_group.create_dataset('images', (n_train, resize_images_to,
                                                                resize_images_to, n_channels), dtype='f')
        hdf5labels = hdf5_train_group.create_dataset('labels', (n_train, 1), dtype='int32')

        train_dataset,train_labels = process_dataset_with_multiprocessing(train_dataset, train_labels, resize_images_to)
        hdf5images[:, :, :, :] = train_dataset
        hdf5labels[:,0] = train_labels[:,0]

        del train_dataset,train_labels

    # ================================================================================================

    # =====================================================================================
    # Saving Testing Data
    with open(test_filename,'rb') as f:
        save = pickle.load(f,encoding="latin1")
        test_dataset, test_labels = np.asarray(save['data']),np.asarray(save['fine_labels']).reshape(-1,)

        # Reshaping images from 1-D to 3-D and converting labels to ndarray
        test_dataset = test_dataset.reshape((-1, original_image_w, original_image_w, n_channels), order='F').astype(np.uint8)
        test_dataset = np.transpose(test_dataset, [0, 2, 1, 3])
        test_dataset -= np.mean(test_dataset, axis=(1, 2, 3))
        test_dataset /= np.std(test_dataset, axis=(1, 2, 3))
        test_labels = test_labels.reshape(-1, )
    # =====================================================================
    # Saving Testing Data to HDF5
    # =====================================================================
    if not ('test' + hdf5_sep + 'images' in dataset_names and 'test' + hdf5_sep + 'labels' in dataset_names):
        hdf5_train_group = hdf5_file.create_group('test')
        hdf5_test_images = hdf5_train_group.create_dataset('images', (n_test, resize_images_to,
                                                                resize_images_to, n_channels), dtype='f')
        hdf5_test_labels = hdf5_train_group.create_dataset('labels', (n_test, 1), dtype='int32')

        test_dataset, test_labels = process_dataset_with_multiprocessing(test_dataset,test_labels, resize_images_to)
        hdf5_test_images[:, :, :, :] = test_dataset
        hdf5_test_labels[:, 0] = test_labels[:,0]


def preprocess_images_by_one(uint8_image, label, resize_to):
    '''
    Normalize images to zero mean and unit variance
    :param img_uint8_batch:
    :return:
    '''

    im = Image.fromarray(uint8_image[0])
    x = np.random.randint(0,8)
    y = np.random.randint(0,8)
    im = im.crop((x, y, x + resize_to, y + resize_to))
    img = np.asarray(im).astype('float32')
    img -= np.mean(img)
    img /= np.std(img)

    assert abs(np.mean(img))<0.01, "Mean (%.3f) for image is not 0"%np.mean(img)
    assert abs(np.std(img))<1.1 and abs(np.std(img))>0.5, "Standard deviation (%.3f) for image is too small or large"%np.std(img)

    return img,label

if __name__=='__main__':

    cifar100_data_dir = 'data'+os.sep+'cifar-100-python'
    cifar10_data_dir = 'data' + os.sep + 'cifar-10'

    save_dir = 'data'
    save_cifar10_as_hdf5(cifar10_data_dir,32,24,save_dir)
    #save_cifar100_as_hdf5(cifar100_data_dir,32,24,save_dir)