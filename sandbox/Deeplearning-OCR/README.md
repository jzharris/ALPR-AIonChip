# Deep Learning OCR
This project is taken from [the following repository](https://github.com/vinayakkailas/Deeplearning-OCR), which is an improvement of [a Tensorflow CRNN implementation](https://github.com/MaybeShewill-CV/CRNN_Tensorflow). For reference, the original paper the CRNN was taken from is http://arxiv.org/abs/1507.05717.

The steps for using this repository are included below. The repository already has some data and a trained model pre-loaded. To quickly test the model accuracy, skip to the testing section.

## Importing the dataset

The dataset must be placed in ./dataset under the Train and Test directories. The directories already have some license plate images from the first dataset included as examples; the script to load these images is at the bottom of ../../main/dataset/import_data.ipynb.

To add your own images, just place them in Train and Test. In each directory, put a file called "sample.txt". In each row, put it in the format of "(Image Name) (Image Label)".

Once the data has been placed in the Train/Test directories, you need to convert the data into tfrecords using the following command:
```
python tools/write_text_features --dataset_dir path/to/your/dataset --save_dir path/to/tfrecords_dir --charset_dir path/to/charset_dir
```

## Training the dataset
After placing the data in its respective directories, you can begin training from scratch, or continuing off an existing trained model.

To train from scratch, use the following command:
```
python tools/train_shadownet.py --dataset_dir dataset
```

To continue training from an existing model, add the "--weights_path" argument with the path to the model:
```
python tools/train_shadownet.py --dataset_dir dataset --weights_path model/shadownet/shadownet_2019-02-14-03-49-38.ckpt-6076
```

To find the path to a trained model, simply look in ./model/shadownet. It includes a pre-trained model on the data from ./dataset.

## Testing the dataset

To test on a trained model, simply call the following command:
```
python tools/test_shadownet.py --dataset_dir dataset/ --weights_path model/shadownet/shadownet_2019-02-14-03-49-38.ckpt-6076
```
