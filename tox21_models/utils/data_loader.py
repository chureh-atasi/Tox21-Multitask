"""Interface to create and train new models"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf

def create_dataset(df, regression, fp_headers):
    """Create dataset for training
    
    Args:
        regression (bool):  
            Optional. Whether a regression or classification model is being  
            trained. Default is True.   

    Return a dataset for tensorflow
    """
    batch_size = 200
    fps = np.stack(df[fp_headers].values).astype(np.float32)
    selector = np.stack(df.selector.values).astype(np.float32)
    # np.newaxis adds one dimension
    if regression:
        target = df.scaled_value.values.astype(np.float32)[:, np.newaxis]
    else:
        target = np.stack(df.class_vals.values).astype(np.float32)
    # Creates dataset
    dataset = tf.data.Dataset.from_tensor_slices(({'selector': selector, 
        'fps': fps, 'prop': df.prop}, target)
    )
    # Caches data and will autotune prefetching to increase training time
    dataset = dataset.cache().batch(batch_size).prefetch(
            tf.data.experimental.AUTOTUNE
    )
    return dataset

def create_dataset_dict(train_df, val_df, props, fp_headers, regression): 
    """Scales train and value properties and creates tf datasets

    Args:
        fp_headers (list):
            List of fingrprint headers
        regression (bool):  
            Optional. Whether a regression or classification model is being  
            trained. Default is True.   

    Returns:
        dict with train dataset, val dataset, and the scaler used to transform
        the data. {'train': train_dataset, 'val': val_dataset, 
        property_scaler': property_scalers}
    """
    scaler = MaxAbsScaler()
    for prop in props:
        train_df[fp_headers] = scaler.fit_transform(
                train_df[fp_headers].to_numpy()
        )
        val_df[fp_headers] = scaler.transform(
                val_df[fp_headers].to_numpy()
        )

    return {'train': create_dataset(train_df, regression, fp_headers), 
            'val': create_dataset(val_df, regression, fp_headers),
            'scaler': scaler}

def create_datasets(df, props, fp_headers, regression=True):
    """Creates five-fold crossvalidation and final datasets

    Args:
        df (pd.DataFrame):  
            Contains smiles, assay type, and value 
        props (list):
            Unique properties to test
        fp_headers (list):
            List of fingrprint headers
        regression (bool):  
            Optional. Whether a regression or classification model is being  
            trained. Default is True.   

    Returns:
        Datasets containing five fold cross validation set, the ensemble
        dataset, the test dataset, and the selector label_binarizer.
    """

    # shuffle dataframe
    df = df.sample(frac=1)


    # Selector is one hot encoding for property list
    sel_LB = LabelBinarizer()
    selectors = sel_LB.fit_transform(df['prop'].values)
    # Need to pass array of single values for making dataset from tensor slices
    df['selector'] = selectors

    # class_vals is one hot encoding for possible classes for val list
    class_LB = LabelBinarizer()
    vals = class_LB.fit_transform(df['CHANNEL_OUTCOME'].values)
    df['class_vals'] = [val for val in vals]

    # Make K fold with preserved percentage of samples for each class
    skf = StratifiedKFold(n_splits=5)
    # Should only do scailing on training dataset, then fit it to test or 
    # validation
    training_df, test_df = train_test_split(df, test_size=0.1, 
            stratify=df.prop, random_state=123)
    training_df, test_df = training_df.copy(), test_df.copy()
    training_df, validation_df = train_test_split(training_df, test_size=0.2, 
            stratify=training_df.prop, random_state=123)
    training_df, validation_df = training_df.copy(), validation_df.copy()


    datasets = []
    # Create the five-fold CV datasets 
    for train_index, val_index in skf.split(training_df, training_df.prop):
        train_df = training_df.iloc[train_index].copy()
        val_df = training_df.iloc[val_index].copy()
        datasets.append(create_dataset_dict(train_df, val_df, props, fp_headers,
            regression)
        )
    
    dataset_final = create_dataset_dict(training_df, validation_df, props, 
            fp_headers, regression
    )

    # Scale to appropriate value for later
    scaler = dataset_final['scaler']
    test_df[fp_headers] = scaler.transform(test_df[fp_headers].values)
    test_dataset = create_dataset(test_df, regression)

    return datasets, dataset_final, test_dataset, sel_LB, class_LB
