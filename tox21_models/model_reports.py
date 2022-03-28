import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def get_data(df):
    pred = []
    actual = []
    for index, row in df.iterrows():
        if (row['pred_col0'] > row['pred_col2']  and row['pred_col0'] > row['pred_col1']):
            pred.append('agonist')
        elif (row['pred_col1'] > row['pred_col0'] and row['pred_col1'] > row['pred_col2']):
            pred.append('inactive')
        elif (row['pred_col2'] > row['pred_col0'] and row['pred_col2'] > row['pred_col1']):
            pred.append('antagonist')
        if (row['is_col0'] == 1):
            actual.append('agonist')
        elif (row['is_col1'] ==1 ):
            actual.append('antagonist')
        elif (row['is_col2'] == 1) :
            actual.append('inactive')
    

    return pred, actual

def build_matrix(df, label):
    pred, actual = get_data(df)
    label_pred = []
    label_actual = []
    for lab in label:
        pred_label = ('pred-' + lab)
        true_label = ('actual-' + lab)
        label_pred.append(pred_label)
        label_actual.append(true_label)

    matrix = confusion_matrix(actual,pred, labels = label)
    print('Confusion matrix : \n',matrix)
    DF = pd.DataFrame(matrix)
    DF = DF.set_axis(label_pred, axis=1, inplace=False)
    DF = DF.set_axis(label_actual, axis=0, inplace=False)
    return DF

def build_df_from_matrix(matrix, label):
    matrix_df = pd.DataFrame()
    row_names = []
    # Want row_headers column first
    for cl in label:
        row_name = f"true_{cl}"
        row_names.append(row_name)
    matrix_df['row_headers'] = row_names
    for j in range(len(classes)):
        cl = classes[j]
        col_name = f"pred_{cl}"
        matrix_df[col_name] = matrix[:, j]
    return matrix_df
