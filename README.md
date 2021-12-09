# mobilenet-v2-custom-dataset
Using Keras MobileNet-v2 model with your custom images dataset

The Keras implementation of MobileNet-v2 (from Keras-Application package) uses by default famous datasets such as imagenet, cifar in a encoded format. But what if we want to use our own custom dataset?

The problem is that if we load all images in a single numpy array, the memory will quickly overload, that's why in this repository we use keras `ImageDataGenerator` class to generate batches during the runtime. The advantage of using `ImageDataGenerator` to generate batches instream of making a hand-made loop over our dataset is that it is directly supported by keras models and we just have to call `fit_generator` method to train on the batches. Moreover, we can easily activate the data augmentation option.  

Our custom dataset need to have the following structure: for every class create a folder containing .jpg sample images:

```
dataset_folder\
    class1\
        image1.jpg
        image2.jpg
    class2\
        image1.jpg
        image2.jpg
        image3.jpg
```


## How to use?

1. Configure the parameters in config.json
2. Train the model using `python train.py`
3. Evaluate the model on test dataset using: `python test.py`
