# Required Libraries and Versions:

* Python: 3.x
* Numpy: 1.12
* Scipy: 0.18
* h5py
* xml.etree.ElementTree
* multiprocessing
* PIL
* six.moves.cPickle
* functools

# What this Repo Do?
It converts several popular datasets into HDF5 format. Currently supports following datasets.

* ILSVRC ImageNet
* CIFAR 10 and CIFAR 100 Datasets
* SVHN Dataset

# How is it Organized?
Code base is pretty simple. It has a single file for each dataset to preprocess data and save as HDF5 (e.g. for Imagenet we have `preprocess_imagenet.py`, CIFAR-10 and CIFAR-100 we have `preprocess_cifar.py` and for SVHN we have `preprocess_svhn.py`). Then I also have several python scripts that can be used to visualize the saved data (e.g. ImageNet - `test_saved_imagenet.py`). Please refer to [my blog post](http://www.thushv.com/computer_vision/bringing-computer-vision-datasets-to-a-single-format-step-towards-consistency/) for a detailed explanation. 

# Accessing and Loading Data Later

You can access this saved data later as:	
```
dataset_file = h5py.File("data" + os.sep + "filename.hdf5", "r")
train_dataset, train_labels = dataset_file['/train/images'], dataset_file['/train/labels']
test_dataset, test_labels = dataset_file['/test/images'], dataset_file['/test/labels']
```

# Tests to Conduct
* Check preprocess_cifar for CIFAR 10 with zero-mean unit-variance - **OK**
* Check preprocess_cifar for CIFAR 100  with zero-mean unit-variance - **OK**
* Check preprocess_cifar for CIFAR 10 with zero-mean - 
* Check preprocess_cifar for CIFAR 100  with zero-mean - 
* Check preprocess_cifar for CIFAR 10 with unit-variance - 
* Check preprocess_cifar for CIFAR 100  with unit-variance - 
* Check preprocess_cifar for CIFAR 10 with Resize - **OK**
* Check preprocess_cifar for CIFAR 100 with Resize - **OK**
* Check SVHN - **OK**
* Check Imagenet from Start - **OK**
* Check Imagenet with crash in middle
* Check `test_saved_cifar` for CIFAR 10 with zero-mean unit-variance - **OK**
* Check `test_saved_cifar` for CIFAR 100 with zero-mean unit-variance - **OK**
* Check `test_saved_cifar` for CIFAR 10 with zero-mean - 
* Check `test_saved_cifar` for CIFAR 100 with zero-mean - 
* Check `test_saved_cifar` for CIFAR 10 with unit-variance -
* Check `test_saved_cifar` for CIFAR 100 with unit-variance -


