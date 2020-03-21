# NinaPro Helper Library
 A helper library to perform essential manipulations on time-series EMG data of NinaPro DB2

# Description

This library aims at easing the preprocessing, amd training of the EMG signals from the NinaPro DB2 database. The working of all the functions is showcased in NinaPro_Utility.ipynb file. 

The library provides following functions :

## get_data()

Download the data from the NinaPro official website and save the data in a folder. The `get_data()` requires the folder path and the file name as the input arguements. 

It returns a dataframe with the EMG signals, labels and the repetition number.

## normalise()

`normalise()` scales the data using the normalise() uses the StandardScaler() from scikit-learn to normalize the data. It fits on training reps only and then transforms the whole data (excluding stimulus and repetition ofcourse). 

Before normalising data :

![](https://github.com/parasgulati8/NinaPro-Helper-Library/blob/master/images/original.JPG)

After **normalization** :

![](https://github.com/parasgulati8/NinaPro-Helper-Library/blob/master/images/normalised.JPG)

**Notice the scale**

## filter_data()

Sometimes, it is required that the signal is filtered with low noise or high noise frequencies. `filter_data` uses Butterworth filter to filter the data. It requires the cutoff frequency, the butterworth order, and the type of filter (btype is one of `lowpass`, `highpass`, `bandpass`). 

The `bandpass` filter requires the `f` value to be a tuple or a list containing lower cutoff frequency and higher cutoff frequency. 

After applying **lowpass** filter:
![](https://github.com/parasgulati8/NinaPro-Helper-Library/blob/master/images/lowpass.JPG)

After applying **bandpass** filter 

![](https://github.com/parasgulati8/NinaPro-Helper-Library/blob/master/images/bandpass.JPG)

After applying **highpass** filter 

![](https://github.com/parasgulati8/NinaPro-Helper-Library/blob/master/images/highpass.JPG)

## rectify()

This function rectifies the signals and converts all the negative values to positive values by simply using the absolute value.

The rectifies signal looks like this :

![](https://github.com/parasgulati8/NinaPro-Helper-Library/blob/master/images/rectified.JPG)

## windowing()

`windowing()` is used to augment the data. The function requires the following arguements : `data`, `reps`, `gestures`, `win_len`, `win_stride`.

`data` = Pandas dataframe just like returned by any of the above functions
`reps` = Repetitions that you want to use for windowing
`gestures` = The gesture movements that you wish to classify
`win_len` = (Length of window in milisecond) x 2. For example, for a window of 300ms, use 600 as the `win_len` since the sampling frequency of signal is 2000Hz.
`win_stride` = (Length of stride in milisecond) x 2. For example, for a stride of 10ms, use 20 as the `win_stride` since the sampling frequency of signal is 2000Hz.

## get_categorical()

For multiclass classification, we need the labels to be represented in one-hot representation.

`get_categorical()` helps in converting the integer labels to one-hot representation.

## plot_cnf_matrix()

It takes the following arguements:
`saved_model` = The model you have already trained
`X_test` 
`y_test`

returns a confision matrix. 
