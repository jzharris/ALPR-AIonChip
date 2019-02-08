# convert the original_dataset into

import os
import os.path as path
import shutil


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

    jpeg_path = path.join(target_path, 'jpeg')
    if not path.isdir(jpeg_path):
        os.mkdir(jpeg_path)

    # do for specific folders:
    search_path = path.join(parent_path, 'Subset_{}'.format(subset), subset, input_set, 'jpeg')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            name = path.splitext(file)
            new_path = '{}_{}.{}'.format(subset, name[0], name[1])

            # save new file to shared folder:
            shutil.copy(path.join(search_path, file), path.join(jpeg_path, new_path))

    xml_path = path.join(target_path, 'xml')
    if not path.isdir(xml_path):
        os.mkdir(xml_path)

    # do for specific folders:
    search_path = path.join(parent_path, 'Subset_{}'.format(subset), subset, input_set, 'jpeg')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            name = path.splitext(file)
            new_path = '{}_{}.{}'.format(subset, name[0], name[1])

            # save new file to shared folder:
            shutil.copy(path.join(search_path, file), path.join(xml_path, new_path))

subsets = [
    'AC', 'LE', 'RP'
]
for subset in subsets:
    convert_files(subset, is_training=True)
    convert_files(subset, is_training=False)
