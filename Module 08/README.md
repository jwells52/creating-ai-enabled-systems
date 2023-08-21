# Module 8 - Design Unsupervised Machine Learning Algorithms

## Objectives
1. Utilize various types of unsupervised machine learning algorithms.
2. Apply unsupervised algorithms to NLP and categorical data.
3. Apply GANs to practical AI problems.

## Assignment 8
The goal of this assignment is to apply K-means clustering to the [Kaggle used car dataset](https://www.kaggle.com/datasets/lepchenkov/usedcarscatalog). Additionally, experiment with different cluster sizes and discribe any insights on the used car market from the clustering experiments.

#### Steps for running the code for this assignment
***From source - NOT RECOMMNEDED - run these commands in your terminal/command prompt***
1. `cd Module 8`
2. `pip install -r requirements.txt`
3. `jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''`
4. In your browser, open `localhost:8888`
5. Open `jwells52_assignment8.ipynb`

***From Docker - RECOMMENDED***
1. `docker pull jwells52/creating-ai-enabled-systems:assignment8`
2. `docker run -it -p 8888:8888 jwells52/creating-ai-enabled-systems:assignment8`
3. In your browser, open `localhost:8888`
4. Open `jwells52_assignment8.ipynb`