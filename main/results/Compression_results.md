# Results of compressing the networks in the pipeline

## YOLOv2 compression

### Huffman encoding
```
Huffman encoding network...
>>>
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 values (39.44 KB) was created
>>>
>>>                       quantized values:   84.99 KB
>>>      the quantized code representation: + 3237.73 KB
>>> bits needed to store quantized network: = 3322.72 KB
>>>
>>>                       bits in codebook:   39.44 KB
>>>          the huff codes representation: + 674.22 KB
>>>      bits needed to store huff network: = 713.65 KB
>>>
>>>                      compression ratio:   4.6559
>>>
```

### Lempel-Ziv-Welch encoding
```
LZ encoding network...
>>>
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 values (39.44 KB) was created
>>>
>>>                       quantized values:   84.99 KB
>>>      the quantized code representation: + 3237.73 KB
>>> bits needed to store quantized network: = 3322.72 KB
>>>
>>>                       bits in codebook:   39.44 KB
>>>          the lziv codes representation: + 652.58 KB
>>>      bits needed to store lziv network: = 692.01 KB
>>>
>>>                      compression ratio:   4.8015
>>>
```

### Huffman encoding on LZ codes
```
LZ encoding network...
>>>
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 values (39.44 KB) was created
>>>
>>>                       quantized values:   84.99 KB
>>>      the quantized code representation: + 3237.73 KB
>>> bits needed to store quantized network: = 3322.72 KB
>>>
>>>                       bits in codebook:   39.44 KB
>>>          the lziv codes representation: + 652.58 KB
>>>      bits needed to store lziv network: = 692.01 KB
>>>
>>>                      compression ratio:   4.8015
>>>
Huffman encoding LZ codes...
>>>
>>> encoded a total of 83 lziv codebooks, and 347652 lziv codes
>>> a total of 83 codebooks containing 137685 values (550.74 KB) was created
>>>
>>>                  bits in lziv codebook:   39.44 KB
>>>          the lziv codes representation: + 652.58 KB
>>>      bits needed to store lziv network: = 692.01 KB
>>>
>>>                  bits in huff codebook:   550.74 KB
>>>          the huff codes representation: + 539.23 KB
>>>      bits needed to store huff network: = 1089.97 KB
>>>
>>>                      compression ratio:   0.6349
>>>
Determining overall compression ratio...
>>>
>>> bits needed to store quantized network:   3322.72 KB
>>>      bits needed to store huff network:   1089.97 KB
>>>                      compression ratio:   3.0485
>>>
```
The compression ratio is very small because the LZ encoded data is very sparse.

### LZ encoding on Huffman codes
```
Huffman encoding network...
>>>
>>> encoded a total of 83 layers, and 3237726 weights
>>> a total of 83 codebooks containing 9859 values (39.44 KB) was created
>>>
>>>                       quantized values:   84.99 KB
>>>      the quantized code representation: + 3237.73 KB
>>> bits needed to store quantized network: = 3322.72 KB
>>>
>>>                       bits in codebook:   39.44 KB
>>>          the huff codes representation: + 674.22 KB
>>>      bits needed to store huff network: = 713.65 KB
>>>
>>>                      compression ratio:   4.6559
>>>
LZ encoding Huffman codes...
>>>
>>> encoded a total of 83 huff codebooks, and 3237726 huff codes
>>> a total of 83 codebooks containing 8851 values (35.40 KB) was created
>>>
>>>                  bits in huff codebook:   39.44 KB
>>>          the huff codes representation: + 674.22 KB
>>>      bits needed to store huff network: = 713.65 KB
>>>
>>>                  bits in lziv codebook:   35.40 KB
>>>          the lziv codes representation: + 589.05 KB
>>>      bits needed to store lziv network: = 624.46 KB
>>>
>>>                      compression ratio:   1.1428
>>>
Determining overall compression ratio...
>>>
>>> bits needed to store quantized network:   3322.72 KB
>>>      bits needed to store lziv network:   624.46 KB
>>>                      compression ratio:   5.3210
>>>
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

### LZ encoding on Huffman codes
```

```