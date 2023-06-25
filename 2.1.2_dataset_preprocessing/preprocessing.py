import argparse
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder, OrdinalEncoder)


def process_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handling outliers in DataFrame. The conditions are set manually.

    Parameters:
        :param df : The DataFrame for processing.

    Returns:
        df : The DataFrame after outliers processing.
    """
    df = df.replace('XNA', np.nan)
    df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
    df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 10, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    df.loc[df['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper=1e6)
    df['REGION_POPULATION_RELATIVE'] = df['REGION_POPULATION_RELATIVE'].clip(upper=0.05)
    df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].clip(upper=70)
    df['CNT_CHILDREN'] = df['CNT_CHILDREN'].clip(upper=6)
    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].clip(upper=7)

    return df


def process_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handling NaNs in DataFrame. The conditions are set manually.

    Parameters:
        :param df : The DataFrame for processing.

    Returns:
        df : The DataFrame after NaNs processing.
    """
    # Information about building
    building_info_cols = [c for c in df.columns if ('_AVG' in c) or ('_MODE' in c) or ('_MEDI' in c)]

    # Information from BCI
    df['FLAG_CREDIT_BUREAU_NAN'] = np.where(df['AMT_REQ_CREDIT_BUREAU_YEAR'].isna(), 1, 0)
    bureau_info_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]

    # Social surroundings with observable 60 DPD
    social_dpd_info = [c for c in df.columns if 'CNT_SOCIAL_CIRCLE' in c]

    # Pensioners & Unemployed
    p_u = (df.NAME_INCOME_TYPE == 'Pensioner') | (df.NAME_INCOME_TYPE == 'Unemployed')
    df.loc[p_u, 'OCCUPATION_TYPE'] = df.loc[p_u, 'OCCUPATION_TYPE'].fillna('XNA')
    df.loc[p_u, 'DAYS_EMPLOYED'] = df.loc[p_u, 'DAYS_EMPLOYED'].fillna(0)
    df.loc[p_u, 'ORGANIZATION_TYPE'] = df.loc[p_u, 'ORGANIZATION_TYPE'].fillna('XNA')

    # The rest of occupation type
    df.OCCUPATION_TYPE = df.OCCUPATION_TYPE.fillna('Working')

    # Own car age
    df.OWN_CAR_AGE = df.OWN_CAR_AGE.fillna(df.OWN_CAR_AGE.mode()[0])

    # Ext_source and cols with small amount of NaNs
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    small_cols = ['AMT_ANNUITY', 'CODE_GENDER', 'CNT_FAM_MEMBERS',
                  'DAYS_LAST_PHONE_CHANGE', 'NAME_TYPE_SUITE']

    # Sum of cols to be filled with mode
    mode = building_info_cols + bureau_info_cols + social_dpd_info + ext_cols + small_cols

    # Filling NaNs with mode
    for col in mode:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def train_test_picker(df: pd.DataFrame, target_feature: str, take_nans: bool) -> pd.DataFrame:
    """
    Splits the dataframe into two parts, one on which the model will be fit,
    the second is the target feature. Allows to filter out NaNs or explicitly take only them.

    Parameters:
        :param df : The DataFrame for processing.
        :param target_feature : Target feature to be predicted.
        :param take_nans [False | True]: Take only NaNs or only NaNs-free data.

    Returns:
        X : train features.
        y : target feature.
    """
    if 'TARGET' in df.columns:
        df = df.drop('TARGET', axis=1)
    
    if take_nans:
        df = df[df[target_feature].isna()]
    else:
        df = df[df[target_feature].notna()]       
    
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]
    
    return X, y


def cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allocates columns of type object and returns their names as a list.

    Parameters:
        :param df : The DataFrame for picking categorial columns.

    Returns:
        cat_cols_list : The list of categorial columns.
    """
    cat_cols_list = df.select_dtypes(include=['object']).columns.tolist()

    return cat_cols_list


def prediction_fill(df, target_feature):
    """
    Uses Сatboost to predict missing values.

    Parameters:
        :param df : The DataFrame with features to train on and to predict.

    Returns:
        df : The whole DataFrame after target_feature predictive filling.
    """
    X, y = train_test_picker(df, target_feature, take_nans=False)
    X_pred, y_pred = train_test_picker(df, target_feature, take_nans=True)

    catboost = CatBoostRegressor(
        iterations=300, loss_function='RMSE',
        learning_rate=0.5, cat_features=cat_cols(X),
        random_state=42, logging_level='Silent')

    catboost.fit(X, y)

    df.loc[y_pred.index, target_feature] = catboost.predict(X_pred)

    return df


def encode_switcher(df: pd.DataFrame, ohe_treshhold: int) -> pd.DataFrame:
    """
    Processes categorical features based on their properties.

    Parameters:
        :param df : The DataFrame for processing.
        :param ohe_treshhold : Number after which the LE will be applied instead of OHE.

    Returns:
        df : The DataFrame after categorial encoding.
    """
    ord_dict = dict(
        NAME_EDUCATION_TYPE=['Lower secondary', 'Secondary / secondary special',
                             'Incomplete higher', 'Higher education', 'Academic degree'],

        WEEKDAY_APPR_PROCESS_START=['MONDAY', 'TUESDAY', 'WEDNESDAY',
                                    'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'],
    )

    for col in cat_cols(df):
        if col in ord_dict.keys():
            df = ord_encoder(df, col, ord_dict)
            
        elif df[col].nunique() > ohe_treshhold:
            df = lab_encoder(df, col)
            
        else:
            df = ohe_encoder(df, col)

    return df


def ohe_encoder(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Encodes categorical variables with Sklearn OneHotEncoder.

    Parameters:
        :param df : The DataFrame for processing.
        :param col : Name of the column to be encoded.

    Returns:
        df : The DataFrame after processing.
    """
    enc = OneHotEncoder(sparse=False, drop='first')

    encoded = pd.DataFrame(
        data=enc.fit_transform(df[col].values.reshape(-1, 1)),
        index=df.index,
        columns=enc.get_feature_names_out([col]))

    df = pd.concat([df, encoded], axis=1).drop(col, axis=1)

    return df


def lab_encoder(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Encodes categorical variables with Sklearn LabelEncoder.

    Parameters:
        :param df : The DataFrame for processing.
        :param col : Name of the column to be encoded.

    Returns:
        df : The DataFrame after processing.
    """
    enc = LabelEncoder()

    encoded = pd.DataFrame(
        data=enc.fit_transform(df[col]),
        index=df.index,
        columns=[f'{col}_le'])

    df = pd.concat([df, encoded], axis=1).drop(col, axis=1)

    return df


def ord_encoder(df: pd.DataFrame, col: str, ord_dict: dict) -> pd.DataFrame:
    """
    Encodes categorical variables with Sklearn OrdinalEncoder.

    Parameters:
        :param df : The DataFrame for processing.
        :param col : Name of the column to be encoded.
        :param ord_dict : Dictionary containing the encoding order.

    Returns:
        df : The DataFrame after processing.
    """
    enc = OrdinalEncoder(categories=[ord_dict[col]])

    encoded = pd.DataFrame(
        data=enc.fit_transform(df[col].values.reshape(-1, 1)),
        index=df.index,
        columns=[f'{col}_ord'])

    df = pd.concat([df, encoded], axis=1).drop(col, axis=1)

    return df


def main(input_filepath: str, output_filepath: str):
    
    feature_to_predict = 'AMT_GOODS_PRICE'
    
    # Read CSV file
    df = pd.read_csv(input_filepath, encoding_errors='ignore', index_col='SK_ID_CURR')
    print(f'Opened {input_filepath}')
    
    # Preprocess
    df = process_outliers(df)
    print('Outliers processing is done!')
    df = process_nans(df)
    print('NaNs processing is done!')
    df = prediction_fill(df, feature_to_predict)
    print(f'{feature_to_predict} predictive filling is done!')
    df = encode_switcher(df, ohe_treshhold=4)
    print('Features encoding is done!')
    
    # Write to CSV file
    df.to_csv(output_filepath, index=True)  # SK_ID_CURR is used as index col on import
    print(f'Written to {output_filepath}')
    

if __name__ == "__main__":
    # Run arguments parser
    parser = argparse.ArgumentParser(description=
        """This script is purposed to fill NaNs, handle outliers 
        and encode features in application_train.csv or application_test.csv 
        | Source: Home Credit Default Risk Kaggle competition 
        (https://www.kaggle.com/c/home-credit-default-risk/data)
        """)
    parser.add_argument('-i', type=str, help=
        """
        Path and filename, I.e. /application_train.csv
        """)
    parser.add_argument('-o', type=str, help=
        """
        Output path and filename, I.e. /application_train.csv
        """)
    args = parser.parse_args()

    # Wrap paths in easy-to-read variables
    input_filepath = args.i
    output_filepath = args.o

    # Run main
    main(input_filepath, output_filepath)
