# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

[//]: # (Image References)
[image1]: ./examples/color_spaces1.png
[image2]: ./examples/color_spaces2.png
[image3]: ./examples/data.png
[image4]: ./examples/hog.png
[image5]: ./examples/combine.png
[image6]: ./examples/windowsearch.png
[image7]: ./examples/findcars.png
[image8]: ./examples/heatmap.png
[Final Video Result]: ./project_video_result.mp4

## Implementation

On this writeup I will describe step by step all the pipeline used on the main IPython notebook located in `./Vehicle Detection.ipynb`.

### 1 - Explore Color Spaces

The first thing I did on this notebook was to explore Color Spaces, by doing this it was easier to identify a color space that might be more suitable to identify cars in a picture. Saturation proved to be one characteristic that most cars shared.

![alt text][image1]
![alt text][image2]

### 2 - Data Exploration

It was time to load the data. First I started with a small sample but finally I noticed that the results improved drastically with the larger dataset.

Basically, I divided the data into to categories, cars and notcars

![alt text][image3]

### 3 - scikit-image HOG

It was time for a quick test of scikit-image HOG. Here is the result for a random car image from the data sample.

![alt text][image4]

### 4 - Combine and Normalize Features

Now, with several feature extraction methods in the toolkit, I was ready to train a classifier. First, as in any machine learning application, I needed to normalize the data. Python's sklearn package provides the StandardScaler() method to accomplish this task. Here is an example of feature extraction before and after the normalization.

![alt text][image5]

### 5 - Color Classify
I used this block to play with different parameters to train a model using only the color feature. It was a lot easier using this notebook to adjust values and iterate quickly. This was the result:

```
Using spatial binning of: 10 and 32 histogram bins
Feature vector length: 396
5.28 Seconds to train SVC...
Test Accuracy of SVC =  0.9032
My SVC predicts:  [ 0.  0.  0.  0.  0.  1.  1.  1.  1.  0.]
For these 10 labels:  [ 0.  1.  0.  0.  0.  1.  1.  1.  1.  0.]
0.00114 Seconds to predict 10 labels with SVC
```

### 6 - HOG Classify
Then I experimented again using only the HOG features and this was the result:

```
42.86 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5292
10.62 Seconds to train SVC...
Test Accuracy of SVC =  0.9764
My SVC predicts:  [ 0.  1.  1.  0.  1.  1.  1.  0.  1.  0.]
For these 10 labels:  [ 0.  1.  1.  0.  1.  1.  1.  0.  1.  0.]
0.00131 Seconds to predict 10 labels with SVC
```

### 7 - Hog Sub-sampling Window Search
This function takes in an image and performs a sliding window search of the normalized combination of features on it. As a result it returns a list of bounding boxes for the search windows, which then I pass to draw draw_boxes() function.

![alt text][image6]

### 8 - Find cars
Now that we have the sliding window search on an image running and a trained classifier  it is time to combine both steps and search for cars!
The key here is the Multi-scale Windows. This allows to search vehicles of any size across the image.

![alt text][image7]

### 9 - Plot heatmap
The idea is to identify Multiple Detections & False Positives.
If the classifier is working well, then the "hot" parts of the map are where the cars are, and by imposing a threshold, you can reject areas affected by false positives.
Here is an example of the output and the heatmap applied to an image:

![alt text][image8]

### 10 - Final Pipeline
The final pipeline is very simple. It takes an image and it looks for cars using the previously trained SVC. Then in creates a heatmap to remove false positives and it adds it to a collection which I use to average the detections. This prevents the square from jumping from frame to frame or from having an frame without a square at all. Finally it draws a blue box on top of the hot window.

### 11 - Apply pipeline to project video
This method extracts each image from the video and process it through the pipeline. As a result we can see the boxes on the final video.

[Final Video Result]
---

### Discussion

For this project, I used a step by step approach which helped me understand each part of the pipeline. This was very helpful when things didn't go exactly as expected. Debugging was a lot easier and concepts became clearer. At the end I put together a pipeline by simply calling the functions I implemented along the way.
I noticed there was a huge difference when I used the larger dataset.

Ideas to improve this project:
* Average the detected boxes by apply the heatmap on more than one frame instead of each individual image.
