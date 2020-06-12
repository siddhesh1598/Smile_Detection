# Smile_Detection

![alt text](https://github.com/siddhesh1598/Smile_Detection/blob/master/thumbnail.jpg?raw=true)


Locating faces in an image and then detecting whether the person is smiling or not. The dataset used is 
[SMILEsmileD](https://github.com/hromi/SMILEsmileD) dataset by **Daniel D. Hromada**. There are 13,165 images in the dataset, where each image has a dimension of 64x64x1 (grayscale). The images in the dataset are tightly cropped around the face.

## Technical Concepts

**LeNet:** The paper can be found [here](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)


## Getting Started

Clone the project repository to your local machine, then follow up with the steps as required.

### Requirements

After cloning the repository, install the necessary requirements for the project.
```
pip install -r requirements.txt
```

### Training

The smiles.hdf5 file is pre-trained in the images from the [SMILEsmileD](https://github.com/hromi/SMILEsmileD). If you wish to train the model from scratch on your own dataset, prepare your dataset in the following way: <br>
1. Load the *smiling* and *not smiling* images into *positive* and *negative* folders respectively. <br>
2. Load these *positive* and *negative* folders into folders *Positive* and *Negative*. Here the *Positive* and *Negative* are our class labels.

So the path for an image belonging to *Positive* label should be: */dataset/Positive/positive/image_01.jpg* <br>
and the path for an image belonging to *Negative* label should be: */dataset/Negative/negative/image_01.jpg*

You can then train the model by using the train_model.py file.

The train_model.py file takes 2 arguments:
1. --dataset: path to the input dataset
2. --model: path to output model

```
python train_model.py --dataset dataset --model resources/smiles.hdf5
```
![alt text](https://github.com/siddhesh1598/Face_Mask_Detection/blob/master/plot.png?raw=true)

The plot for Training and Validation Loss and Accuracy.

### Testing

To test the model on your webcam or any video, use the detect_smile.py file. 
```
python detect_smile.py --cascade resources/haarcascade_frontalface_default.xml --model resources/smiles.hdf5
```

You can pass an optional parameter for the video input:
1. --video: path to the (optional) video file <br>
          *default: face_detector*


## Authors

* **Siddhesh Shinde** - *Initial work* - [SiddheshShinde](https://github.com/siddhesh1598)


## Acknowledgments

* Dataset: [SMILEsmileD](https://github.com/hromi/SMILEsmileD) dataset by [**Daniel D. Hromada**](https://github.com/hromi)
