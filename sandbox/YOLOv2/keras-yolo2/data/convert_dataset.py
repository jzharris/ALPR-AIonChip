# convert the original_dataset into

import os
import os.path as path
import shutil
import fileinput
import cv2
import numpy as np
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm
from scipy.misc import imread


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
    xmls = {}
    search_path = path.join(parent_path, 'Subset_{}'.format(subset), subset, input_set, 'xml')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            name = path.splitext(file)
            xml_dict = xml_to_dict(path.join(search_path, file))
            xml = convert_xml([xml_dict])[0]
            xmls['{}_{}'.format(subset, name[0])] = xml

    search_path = path.join(parent_path, 'Subset_{}'.format(subset), subset, input_set, 'jpeg')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            name = path.splitext(file)
            xml = xmls['{}_{}'.format(subset, name[0])]
            new_path = '{}{}'.format(xml['plate'], name[1])

            # convert image to cropped box using xml path
            if xml is None:
                raise Exception('Should not be None')
            image = cv2.imread(path.join(search_path, file))
            just_lp = img_to_license(image, xml['box'])

            # save new file to shared folder:
            cv2.imwrite(path.join(jpeg_path, new_path), just_lp)
            # shutil.copy(path.join(search_path, file), path.join(jpeg_path, new_path))


def convert_files3(is_training=True):
    if is_training:
        input_sets = ['training']
        output_set = 'train'
    else:
        input_sets = ['testing', 'validation']
        output_set = 'test'

    parent_path = 'ufpr_dataset'

    converted_path = 'converted_ufpr'
    if not path.isdir(path.join(converted_path)):
        os.mkdir(path.join(converted_path))

    target_path = path.join(converted_path, output_set)
    if not path.isdir(target_path):
        os.mkdir(target_path)

    jpeg_path = path.join(target_path, 'jpeg')
    if not path.isdir(jpeg_path):
        os.mkdir(jpeg_path)

    xml_path = path.join(target_path, 'xml')
    if not path.isdir(xml_path):
        os.mkdir(xml_path)

    for input_set in input_sets:

        print('Processing {}:'.format(input_set))

        # copy jpeg file:
        search_path1 = path.join(parent_path, input_set)
        for root1, tracks, _ in os.walk(search_path1):
            for track in tqdm(tracks):
                for root2, _, files in os.walk(path.join(root1, track)):
                    for file in files:
                        name = path.splitext(file)
                        if name[1] == '.png':
                            # copy jpeg file:
                            # print(path.join(root2, file))
                            if not os.path.isfile(path.join(jpeg_path, file)):
                                shutil.copy(path.join(root2, file), path.join(jpeg_path, file))
                        elif name[1] == '.txt':
                            # convert txt file to xml:
                            target_file = path.join(xml_path, '{}.xml'.format(name[0]))
                            if not os.path.isfile(target_file):
                                ######################################################################
                                # variables we need:
                                plate = ''
                                position_plate = ''
                                char_positions = ['', '', '', '', '', '', '']

                                with open(path.join(root2, file)) as txt:
                                    for i, line in enumerate(txt.readlines()):
                                        line = line.replace('\n', '')
                                        if i == 6:
                                            plate = line[6:]
                                        elif i == 7:
                                            position_plate = line[16:]
                                        elif i >= 8:
                                            char_positions[i-8] = line[9:]

                                ######################################################################
                                # preprocess xml objects:
                                boxes = []
                                chars = []
                                for char in plate:
                                    if char != '-' and char != ' ' and type(char) is str:
                                        chars.append(char)

                                with open('bb_xml_ufpr.txt', 'r') as myfile:
                                    bb_template = myfile.read()

                                    # plate position:
                                    position_plate = position_plate.split(' ')
                                    box_left = int(position_plate[0])
                                    box_top = int(position_plate[1])
                                    box_width = int(position_plate[2])
                                    box_height = int(position_plate[3])

                                    x_min = box_left
                                    y_min = box_top
                                    x_max = box_left + box_width
                                    y_max = box_top + box_height

                                    plate_template = bb_template.format(
                                        'plate', x_min, y_min, x_max, y_max
                                    )
                                    boxes.append(plate_template)

                                    # character positions:
                                    for idx, char in enumerate(chars):
                                        position_char = char_positions[idx].split(' ')
                                        box_left = int(position_char[0])
                                        box_top = int(position_char[1])
                                        box_width = int(position_char[2])
                                        box_height = int(position_char[3])

                                        x_min = box_left
                                        y_min = box_top
                                        x_max = box_left + box_width
                                        y_max = box_top + box_height

                                        char_template = bb_template.format(
                                            char, x_min, y_min, x_max, y_max
                                        )
                                        boxes.append(char_template)

                                parsed = ''
                                for box in boxes:
                                    parsed = parsed + box + '\n'

                                ######################################################################
                                # add to main template
                                with open('template_xml_ufpr.txt', 'r') as myfile:
                                    data = myfile.read()
                                    image = imread(path.join(root2, '{}.png'.format(name[0])), mode='L')
                                    formatted = data.format('{}.png'.format(name[0]), image.shape[1], image.shape[0],
                                                            parsed)

                                if not os.path.isfile(target_file):
                                    with open(target_file, 'w+') as xml_file:
                                        xml_file.write(formatted)

            break


subsets = [
    'AC', 'LE', 'RP'
]

convert_files3(is_training=True)
convert_files3(is_training=False)

# for subset in tqdm(subsets):
#     convert_files2(subset, is_training=True)
#     convert_files2(subset, is_training=False)
#
# for subset in tqdm(subsets):
#     convert_files(subset, is_training=True)
#     convert_files(subset, is_training=False)
