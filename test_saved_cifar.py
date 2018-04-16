import h5py
import os
import numpy as np
import scipy.misc
import sys
import getopt

def check_cifar_by_saving_data_to_disk_sorted_by_class(datatype, n_samples, zero_mean, unit_variance):

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
        uint8_img = restore_image_from_standardized(img, zero_mean, unit_variance)
        scipy.misc.imsave(class_dir + os.sep + 'image_%d.jpg'%img_indices_per_label[lbl],uint8_img)
        img_indices_per_label[lbl] += 1

    img_indices_per_label = [0 for _ in range(n_labels)]
    for img, lbl in zip(picked_test_sample_images, picked_test_sample_labels):
        class_dir = test_data_dir + os.sep + str(lbl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        uint8_img = restore_image_from_standardized(img, zero_mean, unit_variance)
        scipy.misc.imsave(class_dir + os.sep + 'image_%d.jpg' % img_indices_per_label[lbl], uint8_img)
        img_indices_per_label[lbl] += 1

def restore_image_from_standardized(img, zero_mean, unit_variance):

    img = img - np.min(img)
    img = img / np.max(img)
    if unit_variance:
        img = img * 255.0
    return img.astype(np.uint8)


if __name__ == '__main__':

    zero_mean = True
    unit_variance = True
    data_type = 'cifar-10'
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "",
            ["data_type=","zero_mean=","unit_variance="])

    except getopt.GetoptError as err:
        print(err.with_traceback())
        print('Please provide the following data to run correctly\n'+
              '--zero_mean=<1 or 0> --unit_variance=<1 or 0>'
            )

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--data_type':
                data_type = str(arg)
            if opt == '--zero_mean':
                zero_mean = bool(int(arg))
            if opt == '--unit_variance':
                unit_variance = bool(int(arg))

    check_cifar_by_saving_data_to_disk_sorted_by_class(data_type,100, zero_mean, unit_variance)