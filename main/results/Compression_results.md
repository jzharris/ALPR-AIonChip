# Results of compressing the networks in the pipeline

## YOLOv2 compression

### Huffman encoding
```
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 32-bit floating point numbers (39.44 KB) was created
>>> a total of 674.22 KB are required to store the encoded variables (8.12 KB per layer on average)
>>> original number of bits needed: 12950.90 KB
>>> new number of bits needed:      713.65 KB
>>> compression ratio:              18.1474
```

### Lempel-Ziv-Welch encoding
```
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 32-bit floating point numbers (39.44 KB) was created
>>> a total of 347.65 KB are required to store the encoded variables (4.19 KB per layer on average)
>>> original number of bits needed: 12950.90 KB
>>> new number of bits needed:      387.09 KB
>>> compression ratio:              33.4573
```

## CRNN compression

