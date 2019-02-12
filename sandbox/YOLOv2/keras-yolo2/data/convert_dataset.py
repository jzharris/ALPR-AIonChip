# convert the original_dataset into

import os
import os.path as path
import shutil
import fileinput
import cv2
import numpy as np
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm


def convert_files(subset, is_training=True):
    if is_training:
        input_set = 'test'
        output_set = 'train'
    else:
        input_set = 'train'
        output_set = 'test'

    parent_path = 'original_dataset'

    converted_path = 'converted_dataset'
    if not path.isdir(path.join(converted_path)):
        os.mkdir(path.join(converted_path))

    target_path = path.join(converted_path, output_set)
    if not path.isdir(target_path):
        os.mkdir(target_path)

    # do for jpegs:
    jpeg_path = path.join(target_path, 'jpeg')
    if not path.isdir(jpeg_path):
        os.mkdir(jpeg_path)

    search_path = path.join(parent_path, 'Subset_{}'.format(subset), subset, input_set, 'jpeg')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            name = path.splitext(file)
            new_path = '{}_{}{}'.format(subset, name[0], name[1])

            # save new file to shared folder:
            shutil.copy(path.join(search_path, file), path.join(jpeg_path, new_path))

    # do for xmls:
    xml_path = path.join(target_path, 'xml')
    if not path.isdir(xml_path):
        os.mkdir(xml_path)

    search_path = path.join(parent_path, 'Subset_{}'.format(subset), subset, input_set, 'xml')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            name = path.splitext(file)
            new_path = '{}_{}{}'.format(subset, name[0], name[1])

            # save new file to shared folder:
            new_file = path.join(xml_path, new_path)
            shutil.copy(path.join(search_path, file), new_file)

            # overwrite new name into xml file...
            with fileinput.FileInput(new_file, inplace=True) as xml:
                for line in xml:
                    print(line.replace('<filename>{}.jpg</filename>'.format(name[0]),
                                       '<filename>{}_{}.jpg</filename>'.format(subset, name[0])), end='')


# Convert dataset 1 into dataset 3 (cropped license plate X, and text y)
def img_to_license(x, bb):
    x1, x2, y1, y2 = bb['xmin'], bb['xmax'], bb['ymin'], bb['ymax']
    pts1 = np.float32([[x1, y1], [x2, y1], [x1, y2]])
    pts2 = np.float32([[0, 0], [x.shape[1], 0], [0, x.shape[0]]])
    return cv2.warpAffine(x, cv2.getAffineTransform(pts1, pts2), (x.shape[1], x.shape[0]))


# Convert XML to nested dictionary
def xml_to_dict(path):
    def xml_to_dict_recursive(node):
        if len(node) == 0:
            return node.text
        return {child.tag:xml_to_dict_recursive(child) for child in node}
    return xml_to_dict_recursive(ElementTree.parse(path).getroot())


def convert_xml(xml_dict_array):
    return [{'plate':xml_dict['object']['platetext'],
             'box':{y:int(xml_dict['object']['bndbox'][y]) for y in xml_dict['object']['bndbox']}} for xml_dict in xml_dict_array]


def convert_files2(subset, is_training=True):
    if is_training:
        input_set = 'test'
        output_set = 'train'
    else:
        input_set = 'train'
        output_set = 'test'

    parent_path = 'original_dataset'

    converted_path = 'converted_dataset2'
    if not path.isdir(path.join(converted_path)):
        os.mkdir(path.join(converted_path))

    target_path = path.join(converted_path, output_set)
    if not path.isdir(target_path):
        os.mkdir(target_path)

    # do for jpegs:
    jpeg_path = path.join(target_path, 'jpeg')
    if not path.isdir(jpeg_path):
        os.mkdir(jpeg_path)

    # get array of bbs corresponding to each jpeg
    bbs = {}
    search_path = path.join(parent_path, 'Subset_{}'.format(subset), subset, input_set, 'xml')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            name = path.splitext(file)
            xml_dict = xml_to_dict(path.join(search_path, file))
            bb = convert_xml([xml_dict])[0]['box']
            bbs['{}_{}'.format(subset, name[0])] = bb
            # name = path.splitext(file)
            # new_path = '{}_{}{}'.format(subset, name[0], name[1])
            #
            # # save new file to shared folder:
            # new_file = path.join(xml_path, new_path)
            # shutil.copy(path.join(search_path, file), new_file)
            #
            # # overwrite new name into xml file...
            # with fileinput.FileInput(new_file, inplace=True) as xml:
            #     for line in xml:
            #         print(line.replace('<filename>{}.jpg</filename>'.format(name[0]),
            #                            '<filename>{}_{}.jpg</filename>'.format(subset, name[0])), end='')

    search_path = path.join(parent_path, 'Subset_{}'.format(subset), subset, input_set, 'jpeg')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            name = path.splitext(file)
            new_path = '{}_{}{}'.format(subset, name[0], name[1])

            # convert image to cropped box using xml path
            bb = bbs['{}_{}'.format(subset, name[0])]
            if bb is None:
                raise Exception('Should not be None')
            image = cv2.imread(path.join(search_path, file))
            just_lp = img_to_license(image, bb)

            # save new file to shared folder:
            cv2.imwrite(path.join(jpeg_path, new_path), just_lp)
            # shutil.copy(path.join(search_path, file), path.join(jpeg_path, new_path))


subsets = [
    'AC', 'LE', 'RP'
]

for subset in tqdm(subsets):
    convert_files2(subset, is_training=True)
    convert_files2(subset, is_training=False)

for subset in tqdm(subsets):
    convert_files(subset, is_training=True)
    convert_files(subset, is_training=False)
