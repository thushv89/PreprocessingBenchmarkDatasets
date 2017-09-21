import h5py
import os
import numpy as np
import scipy.misc

def check_cifar_by_saving_data_to_disk_sorted_by_class(datatype, n_samples):

    train_data_dir = "testing_trainset_images_of_" + datatype
    test_data_dir = "testing_testet_images_of_" + datatype
    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
    if not os.path.exists(test_data_dir):
        os.mkdir(test_data_dir)

    # Datasets
    if datatype == 'cifar-10':
        dataset_file = h5py.File("data" + os.sep + "cifar_10_dataset.hdf5", "r")
        n_labels = 10
    elif datatype == 'cifar-100':
        dataset_file = h5py.File("data" + os.sep + "cifar_100_dataset.hdf5", "r")
        n_labels = 100

    train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']
    test_dataset, test_labels = dataset_file['/test/images'], dataset_file['/test/labels']

    picked_train_sample_images = train_dataset[:n_samples,:,:,:]
    picked_train_sample_labels = train_labels[:n_samples,0]

    picked_test_sample_images = test_dataset[:n_samples, :, :, :]
    picked_test_sample_labels = test_labels[:n_samples, 0]

    del train_dataset,train_labels,test_dataset,test_labels

    img_indices_per_label =  [0 for _ in range(n_labels)]
    for img, lbl in zip(picked_train_sample_images,picked_train_sample_labels):

        class_dir = train_data_dir + os.sep + str(lbl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        uint8_img = restore_image_from_standardized(img)
        scipy.misc.imsave(class_dir + os.sep + 'image_%d.jpg'%img_indices_per_label[lbl],uint8_img)
        img_indices_per_label[lbl] += 1

    img_indices_per_label = [0 for _ in range(n_labels)]
    for img, lbl in zip(picked_test_sample_images, picked_test_sample_labels):
        class_dir = test_data_dir + os.sep + str(lbl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        uint8_img = restore_image_from_standardized(img)
        scipy.misc.imsave(class_dir + os.sep + 'image_%d.jpg' % img_indices_per_label[lbl], uint8_img)
        img_indices_per_label[lbl] += 1

def restore_image_from_standardized(img):

    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255.0
    return img.astype(np.uint8)


if __name__ == '__main__':

    check_cifar_by_saving_data_to_disk_sorted_by_class('cifar-10',100)