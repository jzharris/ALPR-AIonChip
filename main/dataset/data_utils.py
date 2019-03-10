import matplotlib.pyplot as plt
import os
import json
import xml.etree.ElementTree
import scipy.misc
import numpy as np
import pickle
import random
import cv2
import copy

# ---------- #
# Data Utils #
# ---------- #

# Recursively print all folders (to view data format)
def recursive_print(folder_name, tabs=''):
    sub_folders = [x for x in os.listdir(folder_name) if os.path.isdir(folder_name+'/'+x)]
    if len(sub_folders)==0:
        print(tabs+folder_name+": "+str(len([x for x in os.listdir(folder_name)])))
    else:
        for sub_folder in sub_folders:
            print(tabs+sub_folder)
            recursive_print(folder_name+"/"+sub_folder, tabs+"\t")

# Convert XML to nested dictionary
def xml_to_dict(path):
    def xml_to_dict_recursive(node):
        if len(node)==0:
            return node.text
        return {child.tag:xml_to_dict_recursive(child) for child in node}
    return xml_to_dict_recursive(xml.etree.ElementTree.parse(path).getroot())

# Pretty print XML dictionary
def print_xml(xml_dict):
    print(json.dumps(a, indent=4, sort_keys=True))
	
# ---------- #
# Load Data  #
# ---------- #
	
# Load images and labels from both datasets (Returns dictionary of dictionaries)
def load_data(dataset1_path, dataset2_path):    
    def load_images(path, key=lambda x: int(x.split('.')[0])):
        return [scipy.misc.imread(path+"/"+file_name) for file_name in sorted(os.listdir(path),key=key)]
    
    def load_xmls(path):
        return [xml_to_dict(path+"/"+file_name) for file_name in sorted(os.listdir(path),key=lambda x: int(x.split('.')[0]))]
                
    # Convert Images to numpy array
    db_paths = { "ac":dataset1_path+"/Subset_AC/AC/",
                 "le":dataset1_path+"/Subset_LE/LE/",
                 "rp":dataset1_path+"/Subset_RP/RP/" }
    
    dataset1_output = { db_name:
                           {'Xtrain':load_images(db_path+"test/jpeg"),
                            'Xtest':load_images(db_path+"train/jpeg"),
                            'ytrain':load_xmls(db_path+"test/xml"),
                            'ytest':load_xmls(db_path+"train/xml") }
                       for db_name, db_path in db_paths.items()}
    
    dataset2_output = {'X':load_images(dataset2_path, key=lambda x:x),
                       'y':[x.split('.')[0] for x in sorted(os.listdir(dataset2_path),key=lambda x:x)]}
    
    return dataset1_output, dataset2_output
	
# Make modifications on the raw data formats (ex: remove extra XML fields)
def convert_data(raw_data, resized_dimensions = (320, 240)):
    # Not: opencv uses resized_dimensions backwards from numpy (so I used (320, 240) instead of (240, 320))
    def convert_xml(xml_dict_array):
        return [{'plate':xml_dict['object']['platetext'],
                 'box':{y:int(xml_dict['object']['bndbox'][y]) for y in xml_dict['object']['bndbox']}} for xml_dict in xml_dict_array]
    data=dict()
    xtrain, xtest, ytrain, ytest = [], [], [], []
    for data_folder in raw_data: # ac, le, or rp
        xtrain += raw_data[data_folder]['Xtrain']
        xtest += raw_data[data_folder]['Xtest']
        ytrain += convert_xml(raw_data[data_folder]['ytrain'])
        ytest += convert_xml(raw_data[data_folder]['ytest'])
    resized_dimensions = (320, 240) # OPENCV uses it reversed
    for xdata, ydata in [(xtest, ytest), (xtrain, ytrain)]:
        for i in range(len(xdata)):
            resize_ratio = (resized_dimensions[0]/xdata[i].shape[1], resized_dimensions[1]/xdata[i].shape[0])
            xdata[i] = cv2.resize(xdata[i], resized_dimensions)
            ydata[i]['box']['xmin'] *= resize_ratio[0]
            ydata[i]['box']['xmax'] *= resize_ratio[0]
            ydata[i]['box']['ymin'] *= resize_ratio[1]
            ydata[i]['box']['ymax'] *= resize_ratio[1]
    return np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)

# Convert dataset 1 into dataset 3 (cropped license plate X, and text y)
def convert_data2(x, y, resized_dimensions = (320, 240)):
    def img_to_license(x, y_data):
        x1, x2, y1, y2 = y_data['xmin'], y_data['xmax'], y_data['ymin'], y_data['ymax']
        pts1 = np.float32([[x1, y1], [x2, y1], [x1, y2]])
        pts2 = np.float32([[0,0], [x.shape[1], 0], [0, x.shape[0]]])
        return cv2.warpAffine(x, cv2.getAffineTransform(pts1,pts2),(x.shape[1], x.shape[0]))
    return np.array([img_to_license(x[i], y[i]['box']) for i in range(len(y))]), np.array([y[i]['plate'].replace('\n','').replace(' ','') for i in range(len(y))])
	
# --------------- #
# Visualize Data  #
# --------------- #
	
def visualize_dataset(x, y, name="", show_boxes = True, grid_size=2):
    fig = plt.figure()
    fig.suptitle(name)
    for img_num in range(grid_size**2):
        rand_img = random.randint(0,len(x)-1)
        ax = fig.add_subplot(grid_size, grid_size,img_num+1)
        if (show_boxes):
            draw_result(ax, x[rand_img], [y[rand_img]['box']], [y[rand_img]['plate']])
            ax.set_title(y[rand_img]['plate'])
        else:
            ax.imshow(x[rand_img],interpolation='nearest')
            ax.set_title(y[rand_img])

def draw_result(ax, input_img, boxes, plates, certainties=None):
    # make a copy of the img:
    img = copy.deepcopy(input_img)
    
    # should only be one result...bc we are assuming one plate per image
    for i, box in enumerate(boxes):
        xmin = int(box['xmin'])
        xmax = int(box['xmax'])
        ymin = int(box['ymin'])
        ymax = int(box['ymax'])
        
        # set outline shape
        w = (xmax - xmin) // 2
        h = (ymax - ymin) // 2
        x = xmax - w
        y = ymax - h
        
        # set text box width
        min_text_w = 40
        txt_w = w if w > min_text_w else min_text_w
        
        # set font:
        tune_param = 0.9
        fontScale = (img.shape[0] * img.shape[1]) / (1000 * 1000) * tune_param
        
        # draw text box
        textbox_color = (255, 255, 255)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20), (x + txt_w, y - h), textbox_color, -1)
        
        if certainties is None:
            cv2.putText(img, plates[i], (x - w + 3, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, plates[i] + ' : %.2f' % certainties[i], (x - w + 3, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
    # add to matplotlib ax:
    ax.imshow(img, interpolation='nearest')
    ax.set_axis_off()