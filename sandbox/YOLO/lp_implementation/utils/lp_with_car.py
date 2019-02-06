import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg


class lp_with_car(object):
    def __init__(self, phase, rebuild=False):
        self.data_parent = cfg.LP_W_CAR_PATH    # os.path.join(self.devkil_path, 'VOC2007')
        self.data_paths = [os.path.join('Subset_AC', 'AC'), os.path.join('Subset_LE', 'LE'),
                           os.path.join('Subset_RP', 'RP')]
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(list(zip(self.classes, list(range(len(self.classes))))))
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.prepare()

    def get(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 6))     # 6 = changes per class load
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            images[count, :, :, :] = self.image_read(imname)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = os.path.join(self.cache_path, 'lp_w_car_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print(('Loading gt_labels from: ' + cache_file))
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ', self.data_paths)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        gt_labels = []

        # for each LP-parent folder:
        for path in self.data_paths:
            # folder of test/train data
            folder_name = os.path.join(self.data_parent, path, self.phase, 'jpeg')

            # append each image name to gt_labels
            for root, dirs, files in os.walk(folder_name):
                for file in files:
                    imname = os.path.join(folder_name, file)
                    xml_folder = os.path.join(self.data_parent, path, self.phase, 'xml')
                    xmlname = os.path.join(xml_folder, os.path.splitext(file)[0] + '.xml')
                    label, num = self.load_lp_annotation(imname, xmlname)
                    gt_labels.append({'imname': imname, 'label': label})

        print(('Saving gt_labels to: ' + cache_file))
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_lp_annotation(self, imname, xmlname):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 6)) # 6 relates to number of classes....
        filename = xmlname
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)
