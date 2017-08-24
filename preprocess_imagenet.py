__author__ = 'Thushan Ganegedara'

from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from scipy import ndimage
from PIL import Image
import xml.etree.ElementTree as ET
from math import ceil,floor
from scipy.misc import imsave
import h5py
#import _thread as thread
import multiprocessing as mp
from functools import partial
import sys
import time

def save_imagenet_as_memmaps(train_dir,valid_dir, valid_annotation_dir, gloss_fname,
                             n_nat_classes, n_art_classes, resize_images_to, save_dir,
                             n_threads):
    '''
    Retrieves images of imagenet data and store them in a memmap
    :param train_dir: Train data dir (e.g. .../Data/CLS-LOC/train/)
    :param valid_dir: Valid data dir (e.g. .../Data/CLS-LOC/val/)
    :param valid_annotation_dir: Annotation dir (e.g. .../ILSVRC2015/Annotations/CLS-LOC/val/)
    :param n_nat_classes: How many natural object classes in the data
    :param n_art_classes: How many artificial object classes in the data
    :return:
    '''

    dataset_filename = save_dir + os.sep +  'imagenet_250_train_dataset.hdf5'

    gloss_cls_loc_fname = save_dir + os.sep + 'gloss-cls-loc.txt'  # the gloss.txt has way more categories than the 1000 imagenet ones. This files saves the 1000
    selected_gloss_fname = save_dir + os.sep + 'selected-gloss.xml'
    selected_gloss_class_fname = save_dir + os.sep + 'gloss-cls-loc-with-class.xml'
    id_to_label_map_fname = save_dir + os.sep + 'id-to-label-map.xml'

    datasize_fname = save_dir + os.sep + 'dataset_sizes.xml'
    valid_map_fname = save_dir + os.sep + 'temp_valid_fname_to_synset_id_map.pickle'
    label_info_fname = save_dir + os.sep + 'temp_lable_info.pkl'

    hdf5_sep = '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    num_channels = 3

    # label map is needed because I have to create my own class labels
    # my label -> actual label

    # data looks like below
    # n01440764/n01440764_10026 1
    # n01440764/n01440764_10027 2
    # n01440764/n01440764_10029 3

    valid_map = build_or_retrieve_valid_filename_to_synset_id_mapping(valid_annotation_dir, valid_map_fname)

    # ignoring the 0th element because it is just a space
    train_subdirectories = [os.path.split(x[0])[1] for x in os.walk(train_dir)][1:]
    print('Subdirectories: %s\n' % train_subdirectories[:5])
    print('Num subdir: ',len(train_subdirectories))

    if not os.path.exists(gloss_cls_loc_fname):
        synset_id_to_desc_map = write_art_nat_ordered_class_descriptions(train_subdirectories,gloss_fname, gloss_cls_loc_fname)
    else:
        synset_id_to_desc_map = retrieve_art_nat_ordered_class_descriptions(gloss_cls_loc_fname)

    if not os.path.exists(label_info_fname) and not os.path.exists(selected_gloss_fname):
        selected_natural_synsets, selected_artificial_synsets, id_to_label_map = sample_n_natural_and_artificial_classes(n_nat_classes,n_art_classes, gloss_cls_loc_fname)
        print('Summary of selected synsets ...')
        print('\tNatural (%d): %s' % (len(selected_natural_synsets), selected_natural_synsets[:5]))
        print('\tArtificial (%d): %s' % (len(selected_artificial_synsets), selected_artificial_synsets[:5]))

        # check if the class synsets we chose are actually in the training data
        for _ in range(10):
            rand_nat = np.random.choice(selected_natural_synsets)
            rand_art = np.random.choice(selected_artificial_synsets)

            assert rand_nat in train_subdirectories
            assert rand_art in train_subdirectories

        # Temporary file
        with open(label_info_fname, 'wb') as f:
            lable_info = {'nat_synsets':selected_natural_synsets,'art_synsets':selected_artificial_synsets,
                          'id_to_label_map':id_to_label_map}
            pickle.dump(lable_info, f, pickle.HIGHEST_PROTOCOL)

        write_selected_art_nat_synset_ids_and_descriptions(selected_natural_synsets,selected_artificial_synsets,
                                                           synset_id_to_desc_map,selected_gloss_fname,None)
    else:
        with open(label_info_fname, 'rb') as f:
            label_info = pickle.load(f)
            selected_natural_synsets = label_info['nat_synsets']
            selected_artificial_synsets = label_info['art_synsets']
            id_to_label_map = label_info['id_to_label_map']

    # we use a shuffled train set when storing to avoid any order
    n_train = 0
    train_filenames = []
    train_synset_ids = []
    for subdir in train_subdirectories:
        if subdir in selected_natural_synsets or subdir in selected_artificial_synsets:
            file_count = len([file for file in os.listdir(train_dir+os.sep+subdir) if file.endswith('.JPEG')])
            train_filenames.extend([train_dir+os.sep+subdir+os.sep+file for file in os.listdir(train_dir+os.sep+subdir) if file.endswith('.JPEG')])
            train_synset_ids.extend([subdir for _ in range(file_count)])
            n_train += file_count

    train_filenames, train_synset_ids = shuffle_in_unison(train_filenames, train_synset_ids)

    print('Found %d training samples in %d subdirectories...\n' % (n_train, len(train_subdirectories)))
    assert n_train > 0, 'No training samples found'

    # Create memmaps for saving new resized subset of data

    print('Creating train dataset ...')

    if not os.path.exists(dataset_filename):
        hdf5_file = h5py.File(dataset_filename, "w")

    dataset_names = hdf5_file.keys()

    if not ('train'+hdf5_sep+'images' in dataset_names and 'train'+hdf5_sep+'labels' in dataset_names):
        hdf5_train_group = hdf5_file.create_group('train')
        hdf5images = hdf5_train_group.create_dataset('images',(n_train, resize_images_to,
                                                               resize_images_to, num_channels), dtype='f')
        hdf5labels = hdf5_train_group.create_dataset('labels',(n_train,1), dtype='int32')

        filesize_dictionary = dict()
        print("\tMemory allocated for (%d items)..." % n_train)
        filesize_dictionary['train_dataset'] = n_train

        save_train_data_in_filenames(train_filenames, train_synset_ids, hdf5images, hdf5labels,
                                     resize_images_to, num_channels, id_to_label_map, n_threads)

        # TODO: Need to fix this
        #write_dictionary_to_xml(id_to_label_map_fname,id_to_label_map,'synset_id',datatypes[0],['label'],[datatypes[1]])

    else:
        print('Training data exists ...')
        raise NotImplementedError

    write_selected_art_nat_synset_ids_and_descriptions(selected_natural_synsets,selected_artificial_synsets,
                                                       synset_id_to_desc_map,selected_gloss_class_fname,id_to_label_map)

    assert len(id_to_label_map) == n_nat_classes + n_art_classes, \
        'Label map ([synset_id] -> my class label) size does not math class counts'

    # --------------------------------------------------------------------------------
    # Saving validation data
    # --------------------------------------------------------------------------------

    valid_filenames, valid_classes = zip(*valid_map.items())
    print('\tValid filenames:', list(valid_map.keys())[:5])
    print('\tValid classes:', list(valid_map.values())[:5])

    # only get the valid data points related to the classes I picked
    selected_valid_files = []
    selected_valid_synset_ids = []
    for f, c in zip(valid_filenames, valid_classes):
        # only select the ones in the classes I picked
        if c in list(id_to_label_map.keys()):
            fname = f.rstrip('.xml') + '.JPEG'
            selected_valid_files.append(valid_dir+os.sep+fname)
            selected_valid_synset_ids.append(c)

    print('Found %d matching valid files...' % len(selected_valid_files))

    if 'valid'+hdf5_sep+'images' not in dataset_names:

        hdf5_train_group = hdf5_file.create_group('valid')
        hdf5images = hdf5_train_group.create_dataset('images',
                                                     (len(selected_valid_files), resize_images_to, resize_images_to, num_channels),
                                                     dtype='f')
        hdf5labels = hdf5_train_group.create_dataset('labels', (len(selected_valid_files), 1), dtype='int32')

        save_train_data_in_filenames(selected_valid_files,selected_valid_synset_ids,hdf5images,hdf5labels, resize_images_to, num_channels,id_to_label_map, n_threads)

        filesize_dictionary['valid_dataset'] = len(selected_valid_files)
        print('Created tha valid file with %d entries' %filesize_dictionary['valid_dataset'])
    else:
        print('Valid data exists.')
        raise NotImplementedError

    write_dictionary_to_xml(datasize_fname,filesize_dictionary,'dataset',datatypes[0],['size'],[datatypes[1]])


def divide_filenames_for_workers(filenames, synset_ids, n_chunks):
    '''
    Divides a given set of filenames, synset_ids to n_chunks
    :param filenames:
    :param synset_ids:
    :param n_chunks:
    :return:
    '''

    thread_filenames, thread_synset_ids, thread_indices = [],[],[]
    items_per_thread = ceil(len(filenames)*1.0/n_chunks)
    for i in range(n_chunks):
        start_idx = i*items_per_thread
        end_idx = min([(i+1)*items_per_thread,len(filenames)])
        thread_indices.append(list(range(start_idx,end_idx)))
        thread_filenames.append(filenames[start_idx:end_idx])
        thread_synset_ids.append(synset_ids[start_idx:end_idx])
        assert len(thread_indices[-1]) == len(thread_filenames[-1]),\
            'Indices size(%d) and Filename size (%d) does not match'\
            %(len(thread_indices[-1]),len(thread_filenames[-1]))
        assert len(thread_indices[-1]) == len(thread_synset_ids[-1]), 'Indices size and Synset ID size does not match'

    return thread_filenames, thread_synset_ids, thread_indices


def delete_temporary_files(fname_list):
    raise NotImplementedError


def write_dictionary_to_xml(fname,dictionary,key_id,key_datatype, value_id_list,datatype_list):
    '''
    Write a dictionary to XML file.
    If value is a list instead of a single value, you will have to create a list
    of ids for each item in the values of the dictionary. If it is a single value,
     create a list of a single item
    :param fname:
    :param dictionary:
    :param key_id:
    :param value_id:
    :return:
    '''
    root = ET.Element('root')
    for idx,(k,v) in enumerate(dictionary.items()):
        obj = ET.SubElement(root, 'entry')
        ET.SubElement(obj, key_id, {'datatype':key_datatype}).text = k

        if type(v) is list:
            assert len(value_id_list)==len(v),'The value_id_list doesnt match the actual values'
            for v_id,v_dtype, v_item in zip(value_id_list,datatype_list,v):
                ET.SubElement(obj, v_id, {'datatype':v_dtype}).text = str(v_item)

        else:
            assert len(value_id_list)==1,'The value_id_list doesnt match the actual values'
            ET.SubElement(obj, value_id_list[0], {'datatype':datatype_list[0]}).text = str(v)

    tree = ET.ElementTree(root)
    tree.write(fname)


def retrive_dictionary_from_xml(fname):
    '''
    Retrieves a dictionary from the xml file
    :param fname: Name of the xml file
    :return:
    '''
    dictionary = {}
    tree = ET.parse(fname)
    root = tree.getroot()
    for item in root.iter('entry'):
        values = []
        for sub_item in item.iter():
            dtype = sub_item.attrib('datatype')
            if sub_item.tag == 'key':

                 key = get_item_with_dtype(sub_item.text,dtype)
            else:
                 values.append(get_item_with_dtype(sub_item.text,dtype))

        # if the list only has one element
        # dictionary has a single value as value
        if len(values)==1:
            dictionary[key] = values[0]
        else:
            dictionary[key] = values

        break


def shuffle_in_unison(a, b):
    '''
    Shuffle two arrays in the same random order
    :param a: 1st array
    :param b: 2nd array
    :return:
    '''
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

    return a,b

# For the xml file
datatypes = ['string','int32','float32']


def get_item_with_dtype(value,dtype):
    '''
    Get a given string (value) with the given dtype
    :param value:
    :param dtype:
    :return:
    '''
    if dtype==datatypes[0]:
        return str(value)
    elif dtype==datatypes[1]:
        return int(value)
    elif dtype==datatypes[2]:
        return float(value)
    else:
        raise NotImplementedError


def build_or_retrieve_valid_filename_to_synset_id_mapping(valid_annotation_dir, valid_map_fname):
    '''
    Build or retrieve (if exists) a dictionary mapping valid dataset filenames to the corresponding synset ids
    This is important for creating labels for the validation data points
    :param valid_annotation_dir: Annotation directory for valid data
    :param valid_map_fname: The name to save the created dictionary
    :return:
    '''
    print('Building a mapping from valid file name to synset id (also the folder names in the training folder)')
    # valid_map contains a dictionary mapping filename to synset id (e.g. n01440764 is a synset id)
    if not os.path.exists(valid_map_fname):
        valid_map = dict()
        for file in os.listdir(valid_annotation_dir):
            if file.endswith(".xml"):
                head, tail = os.path.split(file)
                tree = ET.parse(valid_annotation_dir + os.sep + file)
                root = tree.getroot()
                for name in root.iter('name'):
                    valid_map[tail] = str(name.text)
                    break
        print('Created a dictionary (valid image file name > synset id )')

        with open(valid_map_fname, 'wb') as f:
            pickle.dump(valid_map, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(valid_map_fname, 'rb') as f:
            valid_map = pickle.load(f)

    return valid_map


def retrieve_art_nat_ordered_class_descriptions(gloss_cls_loc_fname):
    '''
    Retrives the selected natural and artificial synset ids from a previously
    saved file
    :param gloss_cls_loc_fname: Name of the file to retrieve data from
    :return:
    '''

    synset_id_to_description = {}
    with open(gloss_cls_loc_fname, 'r') as f:
        for line in f:
            subdir_str = line.split('\t')[0]
            synset_id_to_description[subdir_str] = line.split('\t')[1]

    return synset_id_to_description


def write_art_nat_ordered_class_descriptions(train_subdirectories,gloss_file_name,gloss_cls_loc_fname):
    '''
    Creates an ordered gloss.txt by the synset ID
    This is important because the first 398 synset IDs are natural items
    and rest is artificial
    :param train_subdirectories:
    :param gloss_file_name: Original gloss.txt
    :param save_dir: Data saving directory
    :return:
    '''

    train_class_descriptions = []
    synset_id_to_description = {}
    with open(gloss_file_name, 'r') as f:
        for line in f:
            subdir_str = line.split('\t')[0]
            if subdir_str in train_subdirectories:
                synset_id_to_description[subdir_str] = line.split('\t')[1]
                train_class_descriptions.append(line)

    with open(gloss_cls_loc_fname, 'w') as f:
        for desc_str in train_class_descriptions:
            f.write(desc_str)

    return synset_id_to_description


def sample_n_natural_and_artificial_classes(n_nat,n_art,gloss_fname):
    '''
    Sample n_nat natural classes and n_art artificial classes from imagenet
    :param n_nat:
    :param n_art:
    :return:
    '''
    natural_synsets, artificial_synsets = [], []
    with open(gloss_fname, 'r') as f:
        for index, line in enumerate(f):
            subdir_str = line.split('\t')[0]
            if index < 398:
                natural_synsets.append(subdir_str)
            else:
                artificial_synsets.append(subdir_str)

    natural_synsets = np.random.permutation(natural_synsets)
    selected_natural_synsets = list(natural_synsets[:n_nat])

    artificial_synsets = np.random.permutation(artificial_synsets)
    selected_artificial_synsets = list(artificial_synsets[:n_art])

    all_selected = selected_natural_synsets + selected_artificial_synsets
    class_labels = range(0,n_nat+n_art)
    synset_id_to_label_map = dict(zip(all_selected,class_labels))
    return selected_natural_synsets,selected_artificial_synsets,synset_id_to_label_map


def write_selected_art_nat_synset_ids_and_descriptions(sel_nat_synsets, sel_art_synsets, synset_id_description_map, filename,label_map=None):
    '''
    Write the selected synset ids to a file
    Good for verifying that there are needed amount of natural and artificlal classes
    :param sel_nat_synsets:
    :param sel_art_synsets:
    :param synset_id_description_map:
    :param filename:
    :param save_dir:
    :return:
    '''
    print(label_map)
    synset_id_dictionary = {} # either synset_id -> ['nat/art','description] or synset_id -> ['description']

    for nat_id in sel_nat_synsets:
        if not label_map:
            synset_id_dictionary[nat_id]= ['nat',synset_id_description_map[str(nat_id)]]
        else:
            synset_id_dictionary[nat_id] = ['nat',label_map[nat_id] , synset_id_description_map[nat_id]]


    for art_id in sel_art_synsets:
        if not label_map:
            synset_id_dictionary[art_id] = ['art',synset_id_description_map[art_id]]
        else:
            synset_id_dictionary[art_id] = ['art',label_map[art_id], synset_id_description_map[art_id]]

    if not label_map:
        write_dictionary_to_xml(filename, synset_id_dictionary,
                                'synset_id', datatypes[0], ['type','description'], [datatypes[0],datatypes[0]])
    else:
        write_dictionary_to_xml(filename,synset_id_dictionary,
                                'synset_id',datatypes[0],
                                ['type','label','description'],[datatypes[0],datatypes[1],datatypes[0]])


def resize_image(fname,resize_to,n_channels):
    '''
    resize image
    if the resize size is more than the actual size, we pad with zeros
    if the image is black and white, we create 3 channels of same data
    if the images has alpha channel, we discard the channel
    :param fname:
    :return:
    '''
    im = Image.open(fname)
    im.thumbnail((resize_to,resize_to), Image.ANTIALIAS)
    resized_img = np.array(im)

    if resized_img.ndim<3:
        resized_img = resized_img.reshape((resized_img.shape[0],resized_img.shape[1],1))
        resized_img = np.repeat(resized_img,3,axis=2)
        assert resized_img.shape[2]==n_channels
    # if there is an alpha layer
    if resized_img.ndim>3:
        resized_img = resized_img[:,:,:3]

    if resized_img.shape[0]<resize_to:
        diff = resize_to - resized_img.shape[0]
        lpad,rpad = floor(float(diff)/2.0),ceil(float(diff)/2.0)
        #print('\tshape of resized img before padding %s'%str(resized_img.shape))
        resized_img = np.pad(resized_img,((lpad,rpad),(0,0),(0,0)),'constant',constant_values=((0,0),(0,0),(0,0)))
        #print('\tshape of resized img after padding %s'%str(resized_img.shape))
    if resized_img.shape[1]<resize_to:
        diff = resize_to - resized_img.shape[1]
        lpad,rpad = floor(float(diff)/2.0),ceil(float(diff)/2.0)
        #print('\tshape of resized img before padding %s'%str(resized_img.shape))
        resized_img = np.pad(resized_img,((0,0),(lpad,rpad),(0,0)),'constant',constant_values=((0,0),(0,0),(0,0)))
        #print('\tshape of resized img after padding %s'%str(resized_img.shape))
    assert resized_img.shape[0]==resize_to
    assert resized_img.shape[1]==resize_to
    assert resized_img.shape[2]==n_channels, 'Resized image as %d channels'%resized_img.shape[2]
    return resized_img


def read_train_data_chunk(fname, synset_id, synset_to_label_map, resize_to, n_channels):
    '''
    Read one example and output the processed image and the label.
    This method is called by the pool workers (multiprocessing)
    :param fname:
    :param synset_id:
    :param synset_to_label_map:
    :param resize_to:
    :param n_channels:
    :return:
    '''
    train_images = np.zeros((resize_to,resize_to,n_channels),dtype=np.float32)
    train_labels = np.zeros((1,), dtype=np.int32)

    resized_img = resize_image(fname, resize_to, n_channels)

    # probably has an alpha layer, ignore these kind of images
    if resized_img.ndim == 3 and resized_img.shape[2] > n_channels:
        print('Ignoring image %s of size %s' % (fname, str(resized_img.shape)))
        return

    # standardizing the image
    resized_img = (resized_img - np.mean(resized_img))
    resized_img /= np.std(resized_img)

    if np.random.random()<0.1:
        assert -.1 < np.mean(resized_img) < .1, 'Mean is not zero'
        assert 0.9 < np.std(resized_img) < 1.1, 'Standard deviation is not one'

    train_images = resized_img
    train_labels[0] = synset_to_label_map[synset_id]

    # helps to track how fast the data is processed

    return train_images, train_labels


def save_train_data_in_filenames(train_filenames, train_synset_ids, hdf5_img, hdf5_labels, resize_to, n_channels, synset_to_label_map, n_chunk):
    '''
    Save the training data using multiprocessing
    This method uses apply_async method.
    Remember than you should pass the hdf5 file in the apply_async(...) because apply_async uses a Queue and
    things in the queue should be picklable
    :param train_dir:
    :param train_filenames:
    :param hdf5_img:
    :param hdf5_labels:
    :param train_offset:
    :param n_channels:
    :return:
    '''

    # divide the full data in to n_chunk sets
    # so it can be performed by multiple workers
    train_filenames_list, train_synsetid_list, train_indices = divide_filenames_for_workers(train_filenames, train_synset_ids, n_chunk)

    # partial function of read_train_data_chunk with only train_filenames, train_synset_ids, train_indices as inputs
    part_pool_func = partial(read_train_data_chunk,
                             resize_to=resize_to, n_channels=n_channels,
                             synset_to_label_map=synset_to_label_map)

    # do not use all the CPUs if there are a lot only use half of them
    # if using all, leave one free
    cpu_count = mp.cpu_count()-1 if mp.cpu_count()<32 else mp.cpu_count()//2
    pool = mp.Pool(cpu_count)
    print('Using %d CPU cores'%cpu_count)

    total_time = 0
    for chunk_id in range(n_chunk):
        t0 = time.time()

        print('Sending Data chunk: ',chunk_id)
        print('\t',len(train_filenames_list[chunk_id]))

        start_idx, end_idx = train_indices[chunk_id][0], train_indices[chunk_id][-1]+1
        print('Reading data from ', start_idx, ' to ', end_idx)
        res_data = pool.starmap(part_pool_func, zip(train_filenames_list[chunk_id],train_synsetid_list[chunk_id])
                                    )
        # get data from after task completion. Not using a timeout as these are expensive operations
        train_images, train_labels = zip(*res_data)

        assert len(train_images) == (end_idx-start_idx), \
            'Train data set size %d and end-start (%d-%d) doesnt match'\
            %(len(train_images),end_idx,start_idx)
        assert len(train_labels) == (end_idx - start_idx), \
            'Train labels size %d and end-start (%d-%d) doesnt match' \
            % (len(train_labels), end_idx, start_idx)

        assert(start_idx < end_idx),'Start index greater than end index'

        # write to the hdf5 file
        hdf5_img[start_idx:end_idx, :, :, :] = np.stack(train_images,axis=0)
        hdf5_labels[start_idx:end_idx, 0] = np.stack(train_labels,axis=0)[:,0]

        t1 = time.time()
        print('Chunk finished %d%% completed (%.3f seconds)'%((chunk_id+1)*100.0/n_chunk,t1-t0))
        total_time += t1 - t0

    # close pools and join to the main process
    pool.close()
    pool.join()

    print('The whole process took %d seconds',total_time)


if __name__ == '__main__':

    train_directory = "/home/tgan4199/imagenet/ILSVRC2015/Data/CLS-LOC/train/"
    valid_directory = "/home/tgan4199/imagenet/ILSVRC2015/Data/CLS-LOC/val/"
    valid_annotation_directory = "/home/tgan4199/imagenet/ILSVRC2015/Annotations/CLS-LOC/val/"
    data_info_directory = "/home/tgan4199/imagenet/ILSVRC2015/ImageSets/"
    data_save_directory = "imagenet_small_test/"
    gloss_fname = '/home/tgan4199/imagenet/ILSVRC2015/gloss_cls-loc.txt'

    save_imagenet_as_memmaps(train_directory,valid_directory,valid_annotation_directory,gloss_fname,125,125,128,data_save_directory,25)