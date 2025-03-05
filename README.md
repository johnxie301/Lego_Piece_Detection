# Obeject Detection Project: Lego_Piece_Detection
Object Detection on Lego pieces using Yolo V11n to train on Lego piece datasets.


## Catalog
1. [Problem](#problem)
2. [Data](#data)
3. [Data Processing](#data-processing)
4. [Model](#model)
5. [Results](#results)

## Problem

The goal of this project is to build an object detection model to identify and classify objects from images filled with random lego pieces. Using dataset provided on kaggle, I will train with a pre-trained object detection model YOLO v11n(You Only Look Once). The challenge involves detecting objects in images and labeling them with the correct class.

## Data

The dataset for this project consists of images and corresponding annotations. The annotations needed to be stored in the **YOLO format**, with a `.txt` file for each image, containing class labels and bounding box coordinates. However, the origional annotations provided was stored in **Pascal VOC format** (XML file). Therefore, we will need to do some data processing before training.  

![demo_1](images/demo_1.jpg)


The dataset is split into a training set and a validation set.

- **Images**: Located in the `datasets/train/images/` and `datasets/val/images/` directories.
- **Labels**: Located in the `datasets/train/labels/` and `datasets/val/labels/` directories in YOLO format.
![dataset_example](images/datasets_example.png)

### Pascal VOC Format
Pascal VOC annotations are typically stored in XML files, with each XML file containing:
- Object class labels
- Bounding box coordinates (xmin, ymin, xmax, ymax)

### Dataset Format (YOLO)
Each label file consists of the following:
<class_id> <x_center> <y_center> <width> <height>

PS:YOLO labels needed to be normalized. 


## Data Processing

### 1. Convert Pascal VOC to YOLO format

To use YOLO, we need to convert the Pascal VOC dataset into **YOLO** format. 

Steps for converting Pascal VOC format to YOLO:
- Use openCV to get the width and Height of each image. 
- Read corresponding XML(Pascal VOC) files.
- Collect their class IDs, total of 200 classes.
- Convert the bounding boxes from Pascal VOC format (absolute pixel coordinates) to the YOLO format(center-relative).
- Create file paths for training, validation and testing.

### 2. Create a .yaml File to Record All the Path and Classes.

5 Variables in it:
-path
-train
-val
-nc (number of classes)
-names(names of all the classes)
![](images/yaml_file_example.png)






