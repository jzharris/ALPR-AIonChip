# Results of pruning and quantization on the End-to-end system

## Test 1
* YOLOv2 with 85.5721% pruned weights
* CRNN with 0% pruned weights

### YOLOv2 accuracies:
* mAP: 0.9848
* Test set boxed: 0 missed LPs
* Train set boxed: 1 missed LPs

### CRNN accuracies:
* Acc (3 or fewer mistakes): 98.367345%
* Acc (2 or fewer mistakes): 93.469387%
* Acc (1 or fewer mistakes): 82.448977%
* Acc (No mistakes):         48.571429%


## Test 2
* YOLOv2 with 2821100 (87.9676%) pruned weights, and 3,206,976 (100%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights

### YOLOv2 accuracies:
* mAP: 0.9897
* Test set boxed: 1 missed LPs (99.7%)
* Train set boxed: - missed LPs (%)

### CRNN accuracies:
* Acc (3 or fewer mistakes): 
* Acc (2 or fewer mistakes): 
* Acc (1 or fewer mistakes): 
* Acc (No mistakes):         