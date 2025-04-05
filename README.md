# Image Classification with Compression and Prediction

This project demonstrates an image classification pipeline using a pre-trained MobileNetV2 model, combined with image compression techniques (RLE, Huffman, and DCT) for processing images before prediction.

## Overview

The notebook `IVP_FA2.ipynb` trains a MobileNetV2 model on a custom dataset of vehicle images (Bus, Car, Motorcycle, Truck) and then applies compression techniques before feeding the processed image to the model for prediction. The process involves:

1.  **Model Training:**
    * Uses a pre-trained MobileNetV2 model and fine-tunes the classifier layer for the 4 vehicle classes.
    * Trains the model on a custom dataset located in `/content/drive/MyDrive/dataset/train`.
    * Evaluates the model on a validation dataset located in `/content/drive/MyDrive/dataset/val`.
2.  **Image Compression:**
    * Implements Run-Length Encoding (RLE), Huffman coding, and Discrete Cosine Transform (DCT) for image compression.
    * Applies these compression techniques to an image fetched from a URL.
3.  **Prediction:**
    * After applying DCT and inverse DCT, the reconstructed image is fed into the trained MobileNetV2 model for classification.
    * The predicted class is then printed.

## Files

* `IVP_FA2.ipynb`: Jupyter notebook containing the code for model training, compression, and prediction.
* `Dataset/train/`: Directory containing the training dataset.
* `Dataset/val/`: Directory containing the validation dataset.
* `Dataset/train/Bus/`: Directory containing Bus images for training.
* `Dataset/train/Car/`: Directory containing Car images for training.
* `Dataset/train/Motorcycle/`: Directory containing Motorcycle images for training.
* `Dataset/train/Truck/`: Directory containing Truck images for training.
* `Dataset/val/Bus/`: Directory containing Bus images for validation.
* `Dataset/val/Motorcycle/`: Directory containing Motorcycle images for validation.
* `Dataset/val/Truck/`: Directory containing Truck images for validation.

## Requirements

* Python 3.x
* PyTorch
* Torchvision
* NumPy
* OpenCV (cv2)
* Requests
* PIL (Pillow)

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy opencv-python requests Pillow
    ```
2.  **Dataset:**
    * Ensure the dataset is located in `/content/drive/MyDrive/dataset/train` and `/content/drive/MyDrive/dataset/val`. If your dataset is in a different location, modify the `train_data` and `val_data` paths in the notebook.
3.  **Run the Notebook:**
    * Open `IVP_FA2.ipynb` in Jupyter Notebook or Google Colab and execute the cells.

## Usage

1.  **Training:**
    * The notebook trains the MobileNetV2 model on the provided dataset.
2.  **Prediction:**
    * The `predict_from_url` function downloads an image from a given URL, compresses it using RLE, Huffman, and DCT, reconstructs it from the DCT output, and then predicts the class using the trained model.
    * The example provided uses this URL : `https://th.bing.com/th/id/OIP.OmszxJcT8NO06xdukAihmwHaE7?w=266&h=180&c=7&r=0&o=5&pid=1.7`
    * You can change the image url to test other images.
3.  **Compression Stats:**
    * The code prints the original size of the image, the size after RLE compression, and the size after Huffman compression.

## Key Functions

* `rle_encode(img)`: Encodes an image using Run-Length Encoding.
* `huffman_encoding(img)`: Encodes an image using Huffman coding.
* `apply_dct(img)`: Applies Discrete Cosine Transform to an image.
* `inverse_dct(dct_img)`: Applies inverse Discrete Cosine Transform to an image.
* `predict_from_url(img_url, model)`: Downloads an image, compresses it, and predicts its class.

## Notes

* The model achieves a validation accuracy of approximately 93.75%.
* The compression methods (RLE, Huffman, DCT) are applied to the grayscale version of the image before feeding it into the model.
* The RLE compression in this case increased the image size. This is because RLE is more effective on images with large runs of same pixel values. Images with high variation of pixel values will result in a larger encoded size.
* The DCT is used to reconstruct the image that is then sent to the model for prediction.
* The warnings about pretrained weights are normal, and are due to the older method being used, the code still functions correctly.
