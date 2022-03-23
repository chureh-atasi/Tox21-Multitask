"""Interface to create and train new models"""
import os
import logging

from pandas.core.base import SelectionMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from kerastuner.tuners import Hyperband
import tensorflow as tf
import joblib

def create_dataset(df, regression, fp_headers):
    """Create dataset for training
    
    Args:
        regression (bool):  
            Optional. Whether a regression or classification model is being  
            trained. Default is True.   

    Return a dataset for tensorflow
    """
    batch_size = 200
    unique_props = list(df.prop.unique())
    features = {}
    for prop in unique_props:
        tdf = df.loc[df.prop == prop]
        features[prop] = np.stack(tdf[fp_headers].values).astype(np.float32)
    fps = np.stack(df[fp_headers].values).astype(np.float32)
    selector = np.stack(df.selector.values).astype(np.float32)
    # np.newaxis adds one dimension
    if regression:
        target = df.scaled_value.values.astype(np.float32)[:, np.newaxis]
    else:
        target = np.stack(df.class_vals.values).astype(np.float32).reshape((-1,1))
    
    # Creates dataset
    #tf.enable_eager_execution()
    #dataset = tf.data.Dataset.from_tensor_slices(({'selector': selector, 
    #    'fps': fps, 'prop': df.prop}, target)
    #)
    dataset = tf.data.Dataset.from_tensor_slices((features, target))
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
    non_fp_headers = [col for col in train_df.columns if col not in fp_headers]
    #train_df[fp_headers] = scaler.fit_transform(
    tdf = pd.DataFrame(scaler.fit_transform(train_df[fp_headers].to_numpy()), 
        columns=fp_headers, index=train_df.index)
    for i in range(len(non_fp_headers) - 1, -1, -1):
        tdf.insert(0, non_fp_headers[i], train_df[non_fp_headers[i]])
    train_df = tdf

    tdf = pd.DataFrame(scaler.transform(val_df[fp_headers].to_numpy()), 
        columns=fp_headers, index=val_df.index)
    for i in range(len(non_fp_headers) - 1, -1, -1):
        tdf.insert(0, non_fp_headers[i], val_df[non_fp_headers[i]])
    val_df = tdf
    logging.info("Done scailing")

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
    logging.info("creating datasets")

    # shuffle dataframe
    df = df.sample(frac=1)

    folder = os.path.dirname(os.path.realpath(__file__))
    # Selector is one hot encoding for property list
    sel_LB = LabelBinarizer()
    selectors = sel_LB.fit_transform(df['prop'].values)
    # Need to pass array of single values for making dataset from tensor slices
    df['selector'] = selectors

    # class_vals is one hot encoding for possible classes for val list
    class_LB = LabelBinarizer()
    vals = class_LB.fit_transform(df['CHANNEL_OUTCOME'].values)
    df['class_vals'] = [val for val in vals]
    #joblib.dump(class_LB, os.path.join(folder, "label_binarizer.pkl"), compress=3)


    # Make K fold with preserved percentage of samples for each class
   
    # Should only do scaling on training dataset, then fit it to test or 
    # validation
    training_df, test_df = train_test_split(df, test_size=0.2, 
            stratify=df.CHANNEL_OUTCOME, random_state=123)
    training_df, test_df = training_df.copy(), test_df.copy() #why
    skf = StratifiedKFold(n_splits=5)
    datasets = []
    # Create the five-fold CV datasets 
    logging.info("Creating cross validation datasets")
    i = 0
    for train_index, val_index in skf.split(training_df, training_df.CHANNEL_OUTCOME):
        logging.info(f"Creating cv dataset {i}")
        i += 1
        train_df = training_df.iloc[train_index].copy()
        val_df = training_df.iloc[val_index].copy()
        datasets.append(create_dataset_dict(train_df, val_df, props, fp_headers,
            regression)
        )
    # may need to find different validation df 
    dataset_final = create_dataset_dict(training_df, test_df, props, 
            fp_headers, regression
    )
    logging.info("Done creating cross validation datasets.")

    # Scale to appropriate value for later
    scaler = dataset_final['scaler']
    test_df[fp_headers] = scaler.transform(test_df[fp_headers].values)
    test_dataset = create_dataset(test_df, regression, fp_headers)
    logging.info("Done creating test dataset.")

    return datasets, dataset_final, test_dataset, sel_LB, class_LB
