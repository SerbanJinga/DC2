# Explanation on the code

In this research project many experiments were done, from different approaches.
This combined with the exploratory approach makes it so that different experiments and functionalities are in very distant parts of the project.
The hope is that this file provides background information about where to find the pertinent information regarding each component.

## Pre-processing

The raw data that was provided by the UK police was stored in `CrimeData/Raw`. It was joint together into many files inside `CrimeData/Processed`.
It also used files in `GeographicalData/Raw` to assign ward information on top of the LSOA information.

The pre-processing is done with methods from file `ProcessData.py`, in particular `join_parsed_dataframes` which uses `parse_dataframe`.
Those methods use `SmallDate.py` and `GeographicConversionHandler.py`.
`SmallDate.py` allows users to manage information and operate with dates.
`GeographicConversionHandler.py` converts LSOA information to wards in an efficient way.

## Basic Analysis

Some of the basic analysis was done by simply taking the processed data from `CrimeData/Processed/Pivot_December_2012_to_march_2023.py` and plotting it in Microsoft Excel.
Other analysis was done in the `Correlation_with_other_crimes.ipynb` and `Monique.ipynb`, such as the correlation matrices.

## Linear Regression

Most of the Linear Regression Analysis can be done in the `LinearRegressionAnalysis.py` file.
First, you can create a dataset to fit the linear regression using `create_convolutional_data`. It was later used to train the Neural networks as well.
Secondly, you can fit the regression to a part of the dataset using `trainLinearModel`.
Lastly, using `experiment_results` you get performance metric and graphs that visualize how the model behaves and its performance.

## Neural Networks

The neural network model is managed in `LSTM/NeuralNetworkPredict.py`,
where the coder can change the model architecture by assigning values to `current_net`. The options are stored in the array `nets`.
The `current_net` changes the model architecture from options provided by the `LSTM/Net.py` file, where architectures are stored.
This was inspired by the template code of data challenge 1, and it helps increase the encapsulation of the program.

The user can also change the batch numbers and epochs from the top of the file.

There are two methods for running the model:
1. `run_single_model` runs a single version of the model, and provides results, good for a first experiment
2. `run_model_battery` runs the model many times (the amount is given by the user as an input) and returns the aggregated results with statistics. It is good to see how the model performs in general and how susceptible to randomness its performance is.

Other files in the `LSTM` folder were previous versions of the code or tests. Similar to the files in `OldCode`.

## `main.py`

The `main.py` file is not important.
It was used to:
1. obtain the benchmarks and other results the `main.py` file was used
2. run tests
3. preprocess the data.
But it was always a temporary placeholder, it is not the center of the code in any way.

## Visualization
There are 3 files that do visualizations:
1. The file `Visualizations/chloropleth_map.py` visualizes how crime is distributed in a given month
2. The file `Visualizations/LSOA_predictions.py` visualizes the predictions of the linear regression stored in `CrimeData/Processed/model_predictions_LSOA.csv`
3. The file `Visualizations/Ward_predictions.py` visualizes the predictions of the linear regression stored in `CrimeData/Processed/model_predictions_Wards.csv`