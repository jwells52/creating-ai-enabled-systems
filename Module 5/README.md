# Programming Assignment 4

### Prerequisites
Create a folder called 'outputs' in a working directory. 
For steps define below please refer to the path of this folder as `/path/to/working/directory/output`

### How to run
1. Run the following command `docker pull <image>` to pull image containing assignment code
2. Run the following command `docker run -it -v /path/to/working/directory/output:/output/`

### Results
You can find results of assignment code in the folder` /path/to/working/directory/output`

The following files that will exist in there after runnning
- manufacturer_name_hists.png - Plots of histograms before and after binning for the `manufacturer_name` column
- manufacturer_name_levels_binned.txt - Text file containing the levels of the `manufacturer_name` column that were binned to a new level labeled 'other'
- model_name_hists.png - Plots of histograms before and after for the `model_name` column
- model_name_levels_binned.txt - Text file containing the levels of the `model_name` column that were binned to a new level labeled 'other'
