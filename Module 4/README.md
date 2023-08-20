# Module 4 - Data Engineering for Unstructured Text

## Objectives
1. Apply data engineering techniques to prepare unstructured text for AI algorithms
2. Apply text preprocessing techniques such as tokenization and word-to-vec


## Assignment
The goal of this programming assignment is to perform text preprocessing techniques such as **tokenization**, **stemming**, and **lemmatization** on the [Amazon musical review data on Kaggle](https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews). 

Note, this assignment specifically requires that a python script and jupyter notebook are submitted. Both the python script and jupyter notebook do the same thing, the difference is that the python script outputs results to folder called `output` and the jupyter notebook has the results printed on in a cell.

#### Steps for running the code for this assignment
***From source - run these commands in your terminal/command prompt***
1. `cd Module 4`
2. `pip install -r requirements.txt`
3. `python jwells52_assignment3.py`
5. Open `output/results.txt` to inspect the results

***From Docker - RECOMMENDED***
1. `mkdir /path/to/jwells52_assignment3/output` !!!NOTE: Set /path/to/ to whatever directory path you desire, make sure that you replace that path in the command on line 3!!!
2. `docker pull jwells52/creating-ai-enabled-systems:assignment3`
3. `docker run -it -p 8888:8888 -v /path/to/jwells52_assignment3/output/:/assignment_code/output/ jwells52/creating-ai-enabled-systems:assignment3`
4. Open `/path/to/jwells52_assignment3/output/results.txt` to inspect the results