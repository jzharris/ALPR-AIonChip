# ECE 209AS AI on Chip

## Final Project

### Running the main project

* ##### Creating the datasets:

    The dataset is generated by running `import_data.ipynb`.
    The LP datasets must be extracted into the `dataset` folder in order for `import_data.ipynb` to read the files.
    The script will generate the `saved datasets` directory, containing the imported LP data.
    

### Running the sandbox/YOLOv2 project

* ##### Download the dataset:

    ##### License Plate segmentation:
    Place the inception pretrained weights in the root folder:
    https://1drv.ms/f/s!ApLdDEW3ut5fec2OzK4S4RpT-SU
    
    Convert the provided dataset by running `convert_dataset.py`. This script will place all LP datasets into one directory, and update their xml `<filname>` references accordingly.
    
    Generate anchors using the following:
    ```
    python gen_anchors.py -c config_lp_bb.json
    # should generate: [1.05,0.87, 1.99,1.46, 2.69,2.30, 2.78,1.82, 3.77,2.83]
    ```
    
    ##### Number segmentation:
    The SVHN VOC xml files are used from:
    https://github.com/penny4860/svhn-voc-annotation-format. Not necessary to run `convert_dataset.py` on these files.
    
    Generate anchors using the following. Copy these anchors into the config file:
    ```
    python gen_anchors.py -c config_num_bb.json
    # should generate: [0.97,5.09, 1.32,9.37, 1.67,7.18, 2.06,10.08, 2.91,10.94]
    ```
    
* ##### Installation:

    Had to install the following due to updated package versions:
    ```
    pip install -U git+https://github.com/apple/coremltools.git # required for using a newer keras version
    ```
    
    Had to install the following due to Windows OS:
    ```
    pip install installation/Shapely-1.6.4.post1-cp35-cp35m-win_amd64.whl # inside project directory
    ```
    
* #### Running:

    ##### License Plate segmentation:

    To train:
    ```
    pyton train.py -c config_lp_seg.json
    ```

    To test:
    ```
    python predict.py -c config_lp_seg.json -w lp_seg_inception.h5 -i images\lp_seg\AC_3.jpg
    python predict.py -c config_lp_seg.json -w lp_seg_inception.h5 -i images\lp_seg\LE_37.jpg
    python predict.py -c config_lp_seg.json -w lp_seg_inception.h5 -i images\lp_seg\RP_32.jpg
    ```

    ##### Number segmentation:

    To train:
    ```
    pyton train.py -c config_num_seg.json
    ```

    To test:
    ```
    python predict.py -c config_num_seg.json -w num_seg_inception.h5 -i images\num_seg\AC_3.jpg
    python predict.py -c config_num_seg.json -w num_seg_inception.h5 -i images\num_seg\LE_37.jpg
    python predict.py -c config_num_seg.json -w num_seg_inception.h5 -i images\num_seg\RP_32.jpg
    ```
    
* #### Debugging:

    Had to perform the fixes to frontend.py according to [this post](https://github.com/experiencor/keras-yolo2/issues/358).
    

* #### Output:

    ![alt_text](https://github.com/jzharris/AIonChip_HOZ/blob/master/sandbox/YOLOv2/keras-yolo2/images/AC_3_detected.jpg)
    ![alt_text](https://github.com/jzharris/AIonChip_HOZ/blob/master/sandbox/YOLOv2/keras-yolo2/images/LE_37_detected.jpg)
    ![alt_text](https://github.com/jzharris/AIonChip_HOZ/blob/master/sandbox/YOLOv2/keras-yolo2/images/RP_32_detected.jpg)


### Running the sandbox/YOLO projects

* ##### Download the dataset:

    Run the `download_data.sh` script in terminal to download the VOC 2007 dataset.
    
* ##### Running:

        python train.py --weights YOLO_small.ckpt --gpu 0
        python test.py --weights YOLO_small.ckpt --gpu 0
    
* ##### Debugging:

    For help debugging, view their [README instructions](https://github.com/jzharris/AIonChip_HOZ/blob/master/sandbox/YOLO/yolo_tensorflow/README.md)
    
* ##### Output:

    ![alt text](https://github.com/jzharris/AIonChip_HOZ/blob/master/sandbox/YOLO/yolo_tensorflow/out/cats.png)
    ![alt text](https://github.com/jzharris/AIonChip_HOZ/blob/master/sandbox/YOLO/yolo_tensorflow/out/person.png)
