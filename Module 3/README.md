# Module 3 - Data Engineering - Computer Vision and Time Series

## Objectives
1. Describe different data engineering techniques related to computer vision and time series data.
2. Apply data engineering techniques to prepare computer vision and time series data for AI algorithms.
3. Setup GitHu and Docker for course assignments.

## Assignment
### Creating GitHub and Docker repo
First, a GitHub and Docker repo are required to be created for holding all the coursework done in this class. If you are seeing this README then it can be assumed you have access to my GitHub repo. The Docker repository that will hold containerized versions of each assignment can be accessed [here](https://hub.docker.com/repository/docker/jwells52/creating-ai-enabled-systems/general). Docker images of every programming assignment in this class will exist in this Docker repo, and each Docker image will have a standard naming convention: `jwells52/creating-ai-enabled-systems:assignmentX` where X is the number of the assignment. For example, this module has programming assignment labelled as Assignment 2, so the Docker image for this assignment is `jwells52/creating-ai-enabled-systems:assigment2`.

### Computer Vision Data Engineering
The goal of the programming assignment is to preprocess image data so that it can be used for a YOLO classifier. The required steps for preprocessing the data are:
1. Resize image to a resolution of 416x416 pixels
2. Normalization each image using a mean=[0.485, 0.456, 0.406] and standard deviation=[0.229, 0.224, 0.225]. Note that the images used in the assignment are in the RGB format (i.e they have 3 channels: Red, Green, and Blue), and the mean and standard deviation for each channel are defined in that order as well.

#### Steps for running the code for this assignment
***From source - run these commands in your terminal/command prompt***
1. `cd Module 3`
2. `pip install -r requirements.txt`
3. `jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''`
4. In your browser, go to `localhost:8888`
5. Open `assignment2.ipynb`

***From Docker - RECOMMENDED***
1. `docker pull jwells52/creating-ai-enabled-systems:assignment2`
2. `docker run -it -p 8888:8888 jwells52/creating-ai-enabled-systems:assignment2`
3. In your browser, go to `localhost:8888`
4. Open `assignment2.ipynb`