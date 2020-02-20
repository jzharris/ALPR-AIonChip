# ECE 209AS AI on Chip

## Final Project

### Pruning explanation - YOLOv2 segmentation

Run `train_prune.py`, calling something similar to `python train_prune.py -c config_lp_seg_mobilenet_prune.json 2>&1 | tee pruned_models/logs.txt`

The following steps are performed:
1. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_prune.py#L142) 
Load pretrained weights (based on `config['train']['pretrained_weights']` from the json file)
2. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_prune.py#L148)
Create a mask (we count any weights that are initially 0 as being pruned from a previous iteration). We use this mask to determine which weights to prune this iteration.
3. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_prune.py#L171)
Recompile the YOLOv2 network with the loaded weights
4. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_prune.py#L190)
Train for `config['train']['nb_epochs']` number of epochs (no pruning, just training). We train the YOLOv2 network with a custom Adam Optimizer that takes the weight mask into account ('freezing' the pruned weights in the mask)
5. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_prune.py#L229)
Perform pruning on the network using [prune_layers](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/prune_network.py#L57) 
function. We add to the mask with any new weights that are zero after this process.
6. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_prune.py#L238)
Save the new weights to a file
7. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_prune.py#L136)
Load the new weights
8. Repeat steps 3 through 7 for `config['train']['train_times']` number of times

### Quantization explanation - YOLOv2 segmentation

Run `train_quantize.py`, calling something similar to `python train_quantize.py -c config_lp_seg_mobilenet_quant.json 2>&1 | tee quant_models/logs.txt`

The following steps are performed:
1. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_quantize.py#L158) 
Load pretrained weights (based on `config['train']['pretrained_weights']` from the json file)
2. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_quantize.py#L163)
Create a mask (we count any weights that are initially 0 as being pruned from a previous iteration). Quantization does not use a mask, but we load one in case the network was pruned previously.
3. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_quantize.py#L186)
Recompile the YOLOv2 network with the loaded weights
4. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_quantize.py#L204)
Train for `config['train']['nb_epochs']` number of epochs (no pruning, just training). We train the YOLOv2 network with a custom Adam Optimizer that takes the weight mask into account ('freezing' the pruned weights in the mask)
5. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_quantize.py#L248)
Perform quantization on the network using [quantize_layers](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/quantize_network.py#L43) 
function
6. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_quantize.py#L254)
Save the new weights to a file
7. [[Ref]](https://github.com/jzharris/AIonChip_HOZ/blob/8e813e824f2357bc2d6422eceeb049377ce88917/main/LP_segmentation/train_quantize.py#L149)
Load the new quantized weights
8. Repeat steps 3 through 7 for `config['train']['train_times']` number of times

### Running the main project

1. ##### Creating the datasets:

    The dataset is generated by running `import_data.ipynb`.
    The LP datasets must be extracted into the `dataset` folder in order for `import_data.ipynb` to read the files.
    The script will generate the `saved datasets` directory, containing the imported LP data.
    
2. ##### TBD

### Running the sandbox/YOLOv2 project

* ##### Download the dataset:

    ##### License Plate localization:
    Place the inception pretrained weights in the root folder:
    https://1drv.ms/f/s!ApLdDEW3ut5fec2OzK4S4RpT-SU
    
    Convert the provided dataset by running `convert_dataset.py`. This script will place all LP datasets into one directory, and update their xml `<filname>` references accordingly.
    
    Generate anchors using the following. Copy these anchors into the config file:
    ```
    python gen_anchors.py -c config_lp_seg.json
    # should generate: [1.05,0.87, 1.99,1.46, 2.69,2.30, 2.78,1.82, 3.77,2.83]
    ```
    
    ##### License Plate reading:
    The bounding boxes are derived from the given dataset by using the process found here:
    https://gurus.pyimagesearch.com/lesson-sample-segmenting-characters-from-license-plates/?fbclid=IwAR1djTQcAUV8Gyi6Oh-7PI-10bYdcFz0_EMmiE5ORpk6H2NVVXVkZ6RaANY
    
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
    python train.py -c config_lp_seg.json
    ```

    To test:
    ```
    python predict.py -c config_lp_seg.json -w lp_seg_inception.h5 -i images\lp_seg\AC_3.jpg
    python predict.py -c config_lp_seg.json -w lp_seg_inception.h5 -i images\lp_seg\LE_37.jpg
    python predict.py -c config_lp_seg.json -w lp_seg_inception.h5 -i images\lp_seg\RP_32.jpg
    ```

    ##### License Plate reading:

    To train:
    ```
    python train.py -c config_char_seg.json
    ```

    To test:
    ```
    python predict.py -c config_char_seg.json -w lp_char_inception.h5 -i images\char_seg\0.jpg
    ```
    
* #### Debugging:

    Had to perform the fixes to frontend.py according to [this post](https://github.com/experiencor/keras-yolo2/issues/358).
    

* #### Output:

    ##### License plate reading
    ![alt_text](https://github.com/jzharris/AIonChip_HOZ/blob/master/sandbox/YOLOv2/keras-yolo2/images/lp_seg/AC_3_detected.jpg)

    ##### Character segmentation
    ![alt_text](https://github.com/jzharris/AIonChip_HOZ/blob/master/sandbox/YOLOv2/keras-yolo2/images/char_seg/2_test_detected.jpg)

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

### TODOs

##### Pipeline item 1: License Plate segmentation
- [x] Train YOLOv2*
    - [x] Initial work with Inception v3 backend
    - [x] Get great resulting bounding boxes
- [x] Create cropped license plate images from output of YOLOv2 network
- [x] Profile the YOLOv2/Inceptionv3 network
    - [x] Profile training phase
    - [x] Profile prediction phase

##### Pipeline item 2: License Plate reading
- [x] Create a VOC-style dataset from original "only lp" dataset using conventional methods**
    - [x] Create intial work (minimum number of converted plates)
    - [x] Convert all license plates in this dataset
- [x] Add image augmentations to improve training
- [x] Train YOLOv2***
    - [x] Initial work with Inception v3 backend
    - [x] Get great resulting bounding boxes
- [x] Profile the YOLOv2/Inceptionv3 network
    - [x] Profile training phase
    - [x] Profile prediction phase
    
##### End-to-end system:
- [x] Use second network (\***) to read plates from the cropped images generated by first network (\*)
- [x] Add bounding boxes from second network into the original image
- [x] Optimize the pipeline
    - [x] Make improvements to bottlenecks found from profiling both networks
    - [x] Investigate combining both networks from the pipeline into a single YOLO network which will perform both
    - [x] Quantize weights
    - [x] Channel pruning using LASSO
    - [x] Huffman encoding
    license plate segmentation and reading

##### Competing methods:
- [x] Implement system for image-to-text using RNN and CTC
    - [x] Compare this method to performance of conventional methods (\**)
    
##### Report:
- [x] Write the report
- [x] Proof read