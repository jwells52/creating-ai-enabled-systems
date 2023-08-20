# Module 5 - Data Engineering for Categorical Data

## Objectives
1. Describe the different types of categorical data.
2. Apply data engineering techniques to prepare categorical data for AI algorithms.
3. Perform operations such as transforming to **nominal data**, **one-hot-encoding**, and **imputation**.

## Assignment 4
The goal of this assignment is to transform the categorical features in the [Kaggle used car dataset](https://www.kaggle.com/datasets/lepchenkov/usedcarscatalog) so that they can be processed by a deep neural network. For this assignment a jupyter notebook or python script can be submitted. I have written both a jupyter notebook and python script, however, it is recommended to use the jupyter notebook for inspecting the results of this assignment, and the steps for running the code will reference the jupyter notebook and **not** the python script.

Another requirement of the assignment is to submit an excel sheet that contains at least two types of data failures, the solutions to fixing them, and advantages/disadvantages to the solutions. The excel sheet I submitted for this assignment can be accessed [here](./jwells52_DataFailureTemplate.xlsx).

#### Steps for running the code for this assignment
***From source - NOT RECOMMNEDED - run these commands in your terminal/command prompt***
1. `cd Module 5`
2. `pip install -r requirements.txt`
3. `jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''`
4. In your browser, open `localhost:8888`
5. Open `jwells52_assignment4.ipynb`

***From Docker - RECOMMENDED***
1. `docker pull jwells52/creating-ai-enabled-systems:assignment4`
2. `docker run -it -p 8888:8888 jwells52/creating-ai-enabled-systems:assignment4`
3. In your browser, open `localhost:8888`
4. Open `jwells52_assignment4.ipynb`