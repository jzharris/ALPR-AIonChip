#!flask/bin/python
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------                                                                                                                             
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end reaches 
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #                                                                                                                                             
#-------------------------------------------------------------------------------------------------------------------------------                                                                                                                              
################################################################################################################################
from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template

from werkzeug.utils import secure_filename
import os
import sys
import random
import re
from tensorflow.python.platform import gfile
from six import iteritems
from tools.demo_shadownet_2 import recognize
import shutil 
import numpy as np

sys.path.append('/home/impadmin/object_detection/models/tf-image-segmentation/tf_image_segmentation')

import tarfile

from datetime import datetime
from scipy import ndimage
from scipy.misc import imsave 
import tensorflow as tf

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
from tensorflow.python.platform import gfile

app = Flask(__name__, static_url_path = "")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



           
#==============================================================================================================================
#                                                                                                                              
#    This function is used to recognize uploaded images.                                                                       
#    used in: dl_image_recognition                                                                                              
#                                                                                                                              
#==============================================================================================================================
@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    print("file upload")
    result = 'static/uploads/test'
    if not gfile.Exists(result):
          os.mkdir(result)
    shutil.rmtree(result)
    if request.method == 'POST' or request.method == 'GET':
        
            file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        
        
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            inputloc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #outputloc = 'out/specific'
            label1, score = recognize(image_path=inputloc, weights_path='model/shadownet_2018-06-22-06-04-22.ckpt-2311')
            image = ndimage.imread(inputloc)
            os.mkdir('static/uploads/test')           
            timestr = datetime.now().strftime("%Y%m%d%H%M%S")
            name= timestr+"."+"test"
            print(name)
            name = 'static/uploads/test/'+name+'.jpg'
            imsave(name, image)
            #os.remove(inputloc)
            label = label1.replace("\n", "")
        
            print("score:", score)

            image_path = "/uploads/test"
            image_list =[os.path.join(image_path,file) for file in os.listdir(result)
                              if not file.startswith('.')]
            print("image name",image_list)
            #if(accuracy < 0.3):
                   #label = 'Unknown Class'
                   #score = '-'
            data = {
                'label':label,
                'score':score,
                'image0':image_list[0]    
            }
            return jsonify(data)
#==============================================================================================================================
#                                                                                                                              
#                                           Main function                                                                        #                                                                        
#                                                                                                                  
#==============================================================================================================================
@app.route("/")
def main():
    
    return render_template("index.html")   
if __name__ == '__main__':
    app.run(debug = True)
