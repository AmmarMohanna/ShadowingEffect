# A CNN Based Algorithm for Shadowing Effect Removal for Target Detection Using FMCW radar

The radar shadowing, or masking, is a well known problem in the radar community. This happens whenever multiple targets are located in the shadowing region of other targets, in which they can be detected with lower reliability. In this paper, a novel solution for the radar shadowing problem is proposed. The solution is based on a CNN model that takes as input the spectrograms obtained after a Short Time Fourier Transform (STFT) analysis and classifies the radar output among two classes: Single or Two targets. The model is based on pre-trained MobileNet. The proposed solution achieves a testing accuracy of 88.7% with a standard deviation of 2.39%. The trained model is considered a light model. It only has 1.06 million parameters, and the inference time using a GPU is 1.64ms.

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

For any help, please contatc me: https://www.linkedin.com/in/ammar-mohanna/

## How to use?

1. Configure the parameters in config.json
2. Train and evaluate the model using `python train.py`
