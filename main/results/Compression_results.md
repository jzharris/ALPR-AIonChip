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
>>> a total of 1390.61 KB are required to store the encoded variables (16.75 KB per layer on average)
>>> original number of bits needed: 12950.90 KB
>>> new number of bits needed:      1430.04 KB
>>> compression ratio:              9.0563
```

### Huffman encoding on LZ codes
```
>>> encoded a total of 83 LZ codebooks, and 347652 LZ codes
>>> a total of 83 codebooks containing 137685 uint32 numbers (550.74 KB) was created
>>> a total of 539.23 KB are required to store the encoded variables (6.50 KB per layer on average)
>>> original number of bits needed to store LZ codes: 1390.61 KB
>>> new number of bits needed:      1089.97 KB
>>> compression ratio:              1.2758
```

## CRNN compression

