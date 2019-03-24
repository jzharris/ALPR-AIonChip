# Results of pruning and quantization on the End-to-end system

## Test 1
* YOLOv2 with 0/3,237,726 (0%) pruned weights, and 0/3,237,726 (0%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights
* CRNN ckpt: model/shadownet/shadownet_2019-03-15-14-12-41.ckpt-46775

### YOLOv2 accuracies:

* mAP: 0.9930
* Train set acc: 1792/1856 = 96.55%
* Test set acc: 306/308 = 99.35%

### CRNN accuracies:
```
Acc (3 or fewer mistakes): 100.000000%
Acc (2 or fewer mistakes): 99.679488%
Acc (1 or fewer mistakes): 98.076922%
Acc (No mistakes):         88.782054%
```


## Test 2
* YOLOv2 with 2,821,100/3,237,726 (87.1321%) pruned weights, and 0/3,237,726 (0%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights
* CRNN ckpt: model/shadownet/shadownet_2019-03-15-14-12-41.ckpt-46775

### YOLOv2 accuracies:
* mAP: 0.9848
* Train set acc: 1806/1856 = 97.31%
* Test set acc: 304/308 = 98.70%

### CRNN accuracies:
```
Acc (3 or fewer mistakes): 100.000000%
Acc (2 or fewer mistakes): 99.022800%
Acc (1 or fewer mistakes): 97.719872%
Acc (No mistakes):         85.993487%
```


## Test 3
* YOLOv2 with 2,821,100/3,237,726 (87.1321%) pruned weights, and 3,206,976/3,237,726 (99%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights
* CRNN ckpt: model/shadownet/shadownet_2019-03-15-14-12-41.ckpt-46775

### YOLOv2 accuracies:
* mAP: 0.9897


## Test 3 (fixed quant)
* YOLOv2 with 2,821,100/3,237,726 (87.1321%) pruned weights, and 3,237,726/3,237,726 (100%) quantized weights
* Augmented CRNN with 0% pruned weights, and 0% quantized weights
* CRNN ckpt: model/shadownet/shadownet_2019-03-15-14-12-41.ckpt-46775

### YOLOv2 accuracies:
* mAP: 0.9857
* Train set acc: 1819/1856 = 98.01%
* Test set acc: 306/308 = 99.35%

### CRNN accuracies:
```
Acc (3 or fewer mistakes): 99.676377%
Acc (2 or fewer mistakes): 99.029124%
Acc (1 or fewer mistakes): 96.440130%
Acc (No mistakes):         87.055016%
```