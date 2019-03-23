# Results of compressing the networks in the pipeline

## YOLOv2 compression

### Huffman encoding
```
Huffman encoding network...
>>>
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 32-bit floating point numbers (39.44 KB) was created
>>> a total of 674.22 KB are required to store the encoded variables (8.12 KB per layer on average)
>>> original number of bits needed: 12950.90 KB
>>> new number of bits needed:      713.65 KB
>>> compression ratio:              18.1474
```

### Lempel-Ziv-Welch encoding
```
LZ encoding network...
>>>
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 32-bit floating point numbers (39.44 KB) was created
>>> a total of 652.58 KB are required to store the encoded variables (7.86 KB per layer on average)
>>> original number of bits needed: 12950.90 KB
>>> new number of bits needed:      692.01 KB
>>> compression ratio:              18.7149
```

### Huffman encoding on LZ codes
```
LZ encoding network...
>>>
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 32-bit floating point numbers (39.44 KB) was created
>>> a total of 652.58 KB are required to store the encoded variables (7.86 KB per layer on average)
>>> original number of bits needed: 12950.90 KB
>>> new number of bits needed:      692.01 KB
>>> compression ratio:              18.7149
Huffman encoding LZ codes...
>>>
>>> encoded a total of 83 LZ codebooks, and 347652 LZ codes
>>> a total of 83 codebooks containing 137685 uint32 numbers (550.74 KB) was created
>>> a total of 539.23 KB are required to store the encoded variables (6.50 KB per layer on average)
>>> original number of bits needed to store LZ codes: 652.58 KB
>>> new number of bits needed:      1089.97 KB
>>> compression ratio:              0.5987
```
The compression ratio is very small because the LZ encoded data is very sparse.

### LZ encoding on Huffman codes
```
Huffman encoding network...
>>>
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 32-bit floating point numbers (39.44 KB) was created
>>> a total of 674.22 KB are required to store the encoded variables (8.12 KB per layer on average)
>>> original number of bits needed: 12950.90 KB
>>> new number of bits needed:      713.65 KB
>>> compression ratio:              18.1474
LZ encoding Huffman codes...
>>>
>>> encoded a total of 83 Huffman codebooks, and 3237726 Huffman codes
>>> a total of 83 codebooks containing 8851 uint32 numbers (35.40 KB) was created
>>> a total of 580.86 KB are required to store the encoded variables (7.00 KB per layer on average)
>>> original number of bits needed to store LZ codes: 6287.88 KB
>>> new number of bits needed:      616.27 KB
>>> compression ratio:              10.2031
```

## CRNN compression

### Huffman encoding
```

```

### Lempel-Ziv-Welch encoding
```

```

### Huffman encoding on LZ codes
```

```