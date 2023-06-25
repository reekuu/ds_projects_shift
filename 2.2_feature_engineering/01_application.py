import argparse
from interest_calc import *
import numpy as np
import pandas as pd


def doc_change_delay(df: pd.DataFrame) -> int:
    """
    Attempts to determine if a passport was issued with a delay.
    The control interval was taken from 31 to 212 days after the birthday.
    It is considered that if the document was issued during this period, then there was a delay.

    Parameters:
        :param df : row of the DataFrame.
    Returns:
        res : 1 if there was delay, 0 otherwise.
    """
    doc_issue_age = df['DAYS_ID_PUBLISH'] - df['DAYS_BIRTH']

    no_penalty_days = 30  
    six_months = 182
    year = 365
  
    days_in45y = 11+year*45 + no_penalty_days + 1
    days_in20y = 4+year*20 + no_penalty_days + 1
    days_in14y = 3+year*14 + no_penalty_days + 1

    range45 = range(days_in45y, days_in45y + six_months)
    range20 = range(days_in20y, days_in20y + six_months)
    range14 = range(days_in14y, days_in14y + six_months)
    
    if doc_issue_age in range45:
        flag = 1
    elif doc_issue_age in range20:
        flag = 1
    elif doc_issue_age in range14:
        flag = 1
    else:
        flag = 0
        
    return flag


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features based on manual preset.

    Parameters:
        :param df : Initial DataFrame.
    Returns:
        res : DataFrame with generated features.
    """
    # 1. DOC_CNT: Number of documents
    cols = [col for col in df.columns if 'FLAG_DOCUMENT_' in col]
    df['DOC_CNT'] = df[cols].sum(axis=1)

    # 2. FLAG_BUILDING_INFO: Is there complete information about the house
    cols = [col for col in df.columns if ('_AVG' in col) or ('_MODE' in col) or ('_MEDI' in col)]
    df['FLAG_BUILDING_INFO'] = np.where(df[cols].isna().sum(axis=1)<30, 1, 0)

    # 3. CLIENT_FULL_AGE: Number of full years
    df['CLIENT_FULL_AGE'] = np.floor(df['DAYS_BIRTH'].abs() / 365).astype('int')

    # 4. DOC_CHANGE_YEAR: Document change year
    df['DOC_CHANGE_YEAR'] = np.floor((df['DAYS_BIRTH'].abs() - df['DAYS_ID_PUBLISH'].abs()) / 365)

    # 5. DOC_CHANGE_DIFF: The difference in time between the change of document and the age at the time of the change of documents
    df['DOC_CHANGE_DIFF'] = df['CLIENT_FULL_AGE'] - df['DOC_CHANGE_YEAR']

    # 6. FLAG_DOC_CHANGE_DELAY: Sign of a delay in changing the document at the age of 14, 20 and 45
    df['FLAG_DOC_CHANGE_DELAY'] = df.apply(doc_change_delay, axis=1)

    # 7. LOAN_INCOME_RATE: The share of money that the client gives for a loan per year
    df['LOAN_INCOME_RATE'] = df['AMT_ANNUITY'] * 12 / df['AMT_INCOME_TOTAL']

    # 8. AVG_CHILDREN_PER_ADULT: Average number of children in a family per adult // +1
    df['AVG_CHILDREN_PER_ADULT'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN'])

    # 9. AVG_INCOME_PER_CHILD: Average income per child
    # WARNING: Probably np.nan is more suitable for replace in terms of using Catboost later.
    df['AVG_INCOME_PER_CHILD'] = (df['AMT_INCOME_TOTAL'] / df['CNT_CHILDREN']).replace(np.inf, 0)

    # 10. AVG_INCOME_PER_ADULT: Average income per adult // +1 if client is not counted as family member
    df['AVG_INCOME_PER_ADULT'] = (df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN'])) 
    df['AVG_INCOME_PER_ADULT'] = df['AVG_INCOME_PER_ADULT'].replace(np.inf, np.nan)

    # 11. AMT_INTEREST_RATE: Approximation of the interest rate calculated using gradient descent
    amt_interest_rate = lambda row: interest_calculator(row['AMT_CREDIT'], row['AMT_ANNUITY'], lr=0.36)
    df['AMT_INTEREST_RATE'] = df.apply(amt_interest_rate, axis=1)

    # 12. EXT_SOURCE_WEIGHTED: Weighted score of external sources
    cols = [col for col in df.columns if 'EXT_SOURCE_' in col]
    source_cnt = df[cols].notna().count(axis=1)
    df['EXT_SOURCE_WEIGHTED'] = (df[cols].sum(axis=1) / source_cnt).replace(np.inf, np.nan)

    ### 13. AVG_GROUP_INCOME_DIFF: the diff between the avg income in the socdem group and the applicant's income
    df['SOCDEM_INCOME'] = df.groupby(['CODE_GENDER', 'NAME_EDUCATION_TYPE'])['AMT_INCOME_TOTAL'].transform('mean')
    df['SOCDEM_INCOME_DIFF'] = df['SOCDEM_INCOME'] - df['AMT_INCOME_TOTAL']
    df = df.drop('SOCDEM_INCOME', axis=1)

    return df


def main(input_filepath: str, output_filepath: str) -> None:
    """
    Generates features from application_train|test.csv and outputs to csv file.

    Parameters:
        :param input_filepath : The filepath to input csv file.
        :param output_filepath : The filepath to output csv file.
    Returns:
        res : Outputs the file with the generated features.
    """     
    # Read CSV file
    df = pd.read_csv(input_filepath, encoding_errors='ignore', index_col='SK_ID_CURR')
    print(f'Opened {input_filepath}')

    # Gererate new features
    df = feature_engineering(df)
    
    # Write to CSV file
    df.to_csv(output_filepath, index=True)  # SK_ID_CURR is used as an index col on import
    print(f'Written to {output_filepath}')
    
    
if __name__ == "__main__":
    # Run arguments parser
    parser = argparse.ArgumentParser(description=
        """This script is purposed for feature generation from 
        application_train.csv or application_test.csv files. 
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
