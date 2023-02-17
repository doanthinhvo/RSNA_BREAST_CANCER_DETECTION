from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
def preprocess_csv_data(df, category_aux_targets, n_folds):

    # ADD StratifiedGroupKFold
    split = StratifiedGroupKFold(n_folds)
    for k, (_, test_idx) in enumerate(split.split(df, df.cancer, groups=df.patient_id)):
        df.loc[test_idx, 'split'] = int(k)
    df.split = df.split.astype(int)    
    
    # Fill na
    df.age.fillna(df.age.mean(), inplace=True)
    df['age'] = pd.qcut(df.age, 10, labels=range(10), retbins=False).astype(int)
    df[category_aux_targets] = df[category_aux_targets].apply(LabelEncoder().fit_transform)

    return df
    

