# Results of pruning and quantization on the End-to-end system

## Test 1
* YOLOv2 with 0/3,237,726 (0%) pruned weights, and 0/3,237,726 (0%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights

### YOLOv2 accuracies:
* mAP: 0.9930
* Test set boxed: 0/300 missed LPs (100% acc)

### CRNN accuracies:
* Acc (3 or fewer mistakes): 99.363059%
* Acc (2 or fewer mistakes): 99.044585%
* Acc (1 or fewer mistakes): 97.770703%
* Acc (No mistakes):         89.171976%


## Test 2
* YOLOv2 with 2,821,100/3,237,726 (87.1321%) pruned weights, and 0/3,237,726 (0%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights

### YOLOv2 accuracies:
* mAP: 0.9848
* Test set boxed: 0/300 missed LPs (100% acc)
* Train set boxed: 2/300 missed LPs (100% acc)

### CRNN accuracies:
* Acc (3 or fewer mistakes): 99.350649%
* Acc (2 or fewer mistakes): 98.376626%
* Acc (1 or fewer mistakes): 97.077924%
* Acc (No mistakes):         85.389608%


## Test 3
* YOLOv2 with 2,821,100/3,237,726 (87.1321%) pruned weights, and 3,206,976/3,237,726 (99%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights

### YOLOv2 accuracies:
* mAP: 0.9897
* Test set boxed: 1/300 missed LPs (99.7% acc)

### CRNN accuracies:
* Acc (3 or fewer mistakes): 99.356914%
* Acc (2 or fewer mistakes): 98.392284%
* Acc (1 or fewer mistakes): 95.819938%
* Acc (No mistakes):         85.852093%


## Test 3 (fixed quant)
* YOLOv2 with 2,821,100/3,237,726 (87.1321%) pruned weights, and 3,237,726/3,237,726 (100%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights

### YOLOv2 accuracies:
* mAP: 0.9857
* Test set boxed: /300 missed LPs (% acc)

### CRNN accuracies:
* 