'''
Script for applying preprocessing techniques to the Used Car Dataset from Kaggle

Preprocessing steps applied are:
1. Frequency Binning
2. Label Encoding
'''
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

########################
### GLOBAL DEFINITIONS
########################
data_path = os.environ['USED_CAR_DATA_PATH']
log_path  = os.environ['LOG_PATH']

########################
# FUNCTION DEFINITIONS
########################
def calc_frequency(series: pd.Series):
    '''
    Function for calculating the frequency of each unique value in the series.
    '''

    # Calculate frequencies of each level
    # Frequency = # of occurences of level / total # of observations
    freqs = {
        level: len(np.where(series == level)[0]) / len(series)
            for level in series.unique()
    }

    return freqs

def bin_by_frequency(series: pd.Series, thresh: float=0.005, verbose: bool=False):
    '''
    Function for binning low occuring levels in a series of categorical data.
    '''
    # Calculate frequency of each unique level
    freqs = calc_frequency(series)

    # Get levels that have a frequency below the specified threshold
    rare_levels = {
        k:v for k,v in freqs.items() if v < thresh
    }

    # Create mapping dictionary to convert the rare occuring levels to a class called 'other'
    other_map = {
        k:'other' for k in rare_levels.keys()
    }

    # Verbose flag for printing data when debugging
    if verbose:
        print('Rare levels found')
        for k,v in rare_levels.items():
            print(k, v)

    return series.replace(other_map)

def get_significant_results(df: pd.DataFrame, df_binned: pd.DataFrame, sig_threshold: int=5):
    '''
    Function for grabbing columns where Frequency Binning resulted in a large amount of levels being binned together.
    '''
    sig_diff_dict = dict()

    print(f'Columns found that had greater than {sig_threshold} levels binned together')
    for _col in cat_cols:
        diff = abs(len(df[_col].unique()) - len(df_binned[_col].unique()))
        diff_set = set(df[_col].unique()).difference(set(df_binned[_col].unique()))
        if diff > sig_threshold:
            print(f'{_col} -> # of levels binned = {diff} ')

            sig_diff_dict[_col] = diff
            
            logfile_path = os.path.join(log_path, f'{_col}_levels_binned.txt')
            with open(logfile_path, 'w+', encoding='utf-8') as _file:
                print(f'column = {_col}', file=_file)
                print('-'*100, file=_file)
                print('\n'.join(list(diff_set)), file=_file)

    return sig_diff_dict

def save_result_histograms(series_before: pd.Series, series_after: pd.Series):
    '''Generate matplotlib histograms of a feature before and after Frequency Binning is applied'''
    
    histplot_path = os.path.join(log_path, f'{col}_hists.png')

    plt.figure(1, figsize=(24, 6))
    plt.title(col)
    
    ax1 = plt.subplot(1, 2, 1)
    series_before.sort_values().hist()
    ax1.set_title('Before Frequency Binning')
    plt.xticks(rotation = 45)

    ax2 = plt.subplot(1, 2, 2)
    series_after.sort_values().hist()
    ax2.set_title('After Frequency Binning')

    plt.xticks(rotation = 45)
    plt.savefig(histplot_path)
    plt.show()
    plt.close()

########################
# CLASS DEFINITIONS
########################
class FrequencyBinning():
    '''
    Wrapper class for applying the frequency binning functions like sklearn transformer classes.
    
    Example:
    ```
    bin = FrequencyBinning(threshold=0.005)
    df_binned = bin.fit_transform(df)
    ```
    '''
    def __init__(self, threshold: float=0.005):
        self.threshold = threshold

    def fit(self, X=None, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        for col in X.columns:
            series = X.loc[:, col]
            X.loc[:, col] = bin_by_frequency(series, self.threshold)
        
        return X
    
    def fit_transform(self, X:pd.DataFrame, y=None):
        return self.fit().transform(X)

if __name__ == '__main__':
    # Load data
    print('Loading data...')
    df = pd.read_csv(data_path)

    # Find categorical columns
    cat_cols  = df.columns[(df.dtypes == object) | (df.dtypes == bool)].values
    ########################################
    # Preprocessing categorical data stage
    # Steps for preprocessing are:
    # 1. Frequency Binning
    # 2. LabelEncoding
    # 3. ?? Standard Scaling??
    ##########################################

    # Step 1: Frequency Binning
    print('Frequency binning...')
    threshold = 0.0025
    df_binned = df.copy()
    bin = FrequencyBinning(threshold=threshold)
    df_binned.loc[:, cat_cols] = bin.fit_transform(df_binned[cat_cols])
    
    # Step 2: Label Encoding
    print('Label encoding...')
    df_encoded = df_binned.copy()
    df_encoded.loc[:, cat_cols] = df_encoded[cat_cols].apply(LabelEncoder().fit_transform)
    
    # Step 3: Standard Scaling
    print('Standard scaling...')
    df_scaled = df_encoded.copy()
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_scaled), columns=df_encoded.columns)
    
    print('Finished preprocessing!')
    
    # Save preprocessed data to a csv file
    df_scaled.to_csv('/output/cars-preprocessed.csv', index=False)

    ##########################################
    # REPORT FINDINGS 
    ##########################################
    
    print('\n\Saving outputs...')

    # Find columns that were had a significant amount of levels reduced by Frequency Binning
    sig_dict = get_significant_results(df, df_binned)
    for col in sig_dict:
        save_result_histograms(df[col], df_binned[col])