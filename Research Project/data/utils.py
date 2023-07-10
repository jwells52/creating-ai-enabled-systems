'''
Module for utility functions regarding data preprocessing
'''

import pandas as pd

from sklearn.model_selection import train_test_split

def generate_train_validation(
        df: pd.DataFrame,
        split=0.2,
        random_state=42,
        threshold=None,
    ):
    '''
    Function for generating csv files that list the classes and images that will be used for
    training and validation sets. Test set is not creating because that is provided by the Kaggle Dataset.
    '''
    if threshold is not None:
        df = df[df['class_count'] > threshold]
    
    y = df['Id'].unique()

    y_train, y_val = train_test_split(y, test_size=split, random_state=random_state)

    train_set = df[df['Id'].isin(y_train)]
    valid_set = df[df['Id'].isin(y_val)]

    train_savepath = f'training_{threshold}samples.csv' if threshold is not None else 'training.csv'
    valid_savepath = f'validation_{threshold}samples.csv' if threshold is not None else 'validation.csv'

    train_set.to_csv(train_savepath, index=False)
    valid_set.to_csv(valid_savepath, index=False)

if __name__ == "__main__":
    images_and_labels = pd.read_csv('train_v2.csv')
    generate_train_validation(images_and_labels, threshold=10)
