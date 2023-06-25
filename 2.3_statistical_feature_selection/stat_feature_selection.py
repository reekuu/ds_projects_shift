import argparse
import numpy as np
import pandas as pd
from scipy.stats import (
    anderson, chi2_contingency, mannwhitneyu, ttest_ind
    )


def is_normal(series: pd.Series) -> bool:
    """
    Based on Anderson-Darling test. Determines if series follows a normal distribution.

    Parameters:
        :param series: The series on which the testing will be performed.
    Returns:
        res : The result is True if series follows a normal distribution, False otherwise.
    """
    # Anderson-Darling test
    stat, crit_val, sign_level = anderson(series, 'norm')

    # crit_val[4] == 5%
    if stat < crit_val[4]:
        # H0 is not rejected
        normalcy = True
    elif stat >= crit_val[4]:
        # H0 is rejected
        normalcy = False

    return normalcy


def students_ttest(label_0: pd.Series, label_1: pd.Series, alpha: float = 0.05) -> bool:
    """
    Based on Student's T-test. Determines if the feature is significant or not.
    Tests the null hypothesis that two independent samples have identical average (expected) values.

    Parameters:
        :param label_0: The series with target label == 0 for the test.
        :param label_1: The series with target label == 1 for the test.
        :param alpha : The alpha for p-value, default is 0.05.
    Returns:
        res : The result is True if feature is significant, False otherwise. 
    """
    # Student's T-test
    _, p_tt = ttest_ind(label_0, label_1)

    if p_tt < alpha:
        # H0 is rejected
        significancy = True
    elif p_tt >= alpha:
        # Failed to reject H0
        significancy = False

    return significancy


def mannwhitneyu_test(label_0: pd.Series, label_1: pd.Series, alpha: float = 0.05) -> bool:
    """
    Based on The Mann-Whitney U test. Determines if the feature is significant or not.
    Tests the null hypothesis that the distribution underlying label_0 is the same
    as the distribution underlying label_1.

    Parameters:
        :param label_0: The series with target label == 0 for the test.
        :param label_1: The series with target label == 1 for the test.
        :param alpha : The alpha for p-value, default is 0.05.
    Returns:
        res : The result is True if feature is significant, False otherwise. 
    """
    # Mann-Whitney U rank test
    _, p_mw = mannwhitneyu(label_0, label_1)

    if p_mw < alpha:
        # H0 is rejected
        significancy = True
    elif p_mw >= alpha:
        # Failed to reject H0
        significancy = False

    return significancy


def chi2_cont_test(df: pd.DataFrame, feature: str, target: str, alpha: float = 0.05) -> bool:
    """
    Based of Chi-square test of independence of variables in a contingency table.
    Determines if the feature is significant or not.

    Parameters:
        :param df: The DataFrame containing the test and target features.
        :param feature: The name of the feature for which the test will performed.
        :param target: The name of the target feature.
        :param alpha : The alpha for p-value, default is 0.05.
    Returns:
        res : The result is True if feature is significant, False otherwise. 
    """
    cross_tab = pd.concat([
        pd.crosstab(df[feature], df[target], margins=False),
        df.groupby(feature)[target].agg(['count', 'mean']).round(4)
    ], axis=1).rename(columns={0: f"label=0", 1: f"label=1", "mean": 'probability_of_default'})

    cross_tab['probability_of_default'] = np.round(cross_tab['probability_of_default']*100, 2)

    # Chi-square contingency test
    chi2_stat, p_chi2, dof, expected = chi2_contingency(cross_tab.values)

    if p_chi2 <= alpha:
        # H0 is rejected
        significancy = True
    elif p_chi2 > alpha:
        # Failed to reject H0
        significancy = False

    return significancy


def is_significant(df: pd.DataFrame, feature: str, target: str = 'TARGET', aplha: float = 0.05) -> bool:
    """
    Determines if the feature is significant or not.
    
    Parameters:
        :param df: DataFrame with two columns: the feature for testing and the target feature.
        :param feature: The name of the feature for testing.
        :param target: The target feature name.
        :param alpha : The alpha for p-value, default is 0.05.
    Returns:
        res : The result is True if feature is significant, False otherwise. 
    """
    label_0 = df.loc[df[target] == 0, feature].dropna()
    label_1 = df.loc[df[target] == 1, feature].dropna()
    
    is_categoriсal = label_0.dtype == 'O'
    is_binary = label_0.nunique() == 2
    
    if is_categoriсal or is_binary:
        significancy = chi2_cont_test(df[[feature, target]], feature, target, aplha)
    
    elif is_normal(label_0) and label_0.shape[0] >= 30:
        significancy = students_ttest(label_0, label_1, aplha)
        
    else:
        significancy = mannwhitneyu_test(label_0, label_1, aplha)
    
    return significancy


def merge_target(df: pd.DataFrame, input_filepath: str) -> pd.DataFrame:
    '''
    Recieves raw imported DataFrame and adds target column.

    Paramaeters:
        :param df : Imported DataFrame.
        :param input_filepath : The filepath to input csv file.
    Returns:
        res : Outputs the DataFrame with merged target column.
    '''
    # Extract file name and path from input_filepath
    file_name = input_filepath.split('/')[-1]
    file_path = '/'.join(input_filepath.split('/')[:-1])

    # List of DataFrames with SK_ID_CURR as Primary key
    sk_id_list = [
        'bureau.csv', 'previous_application.csv',
        'POS_CASH_balance.csv', 'instalments_payments.csv',
        'credit_card_balance.csv'
    ]

    # List of DataFrames with SK_ID_BUREAU as Primary key    
    bureau_id_list = ['bureau_balance.csv']

    # Merge the target feature in the original DataFrame
    if file_name in sk_id_list:
        df = df.merge(pd.read_csv(file_path+'/sk_id_target.csv'))
    elif file_name in bureau_id_list:
        df = df.merge(pd.read_csv(file_path+'/bureau_id_target.csv'))
        
    return df


def main(input_filepath: str, output_filepath: str) -> None:
    """
    Based on the feature type (binary, categorical, normally distributed etc.)
    performs stat test to determine the feature significance.

    Parameters:
        :param input_filepath : The filepath to input csv file.
        :param output_filepath : The filepath to output csv file.
    Returns:
        res : Outputs the file with only the significant features.
    """
    # Read CSV file
    df = pd.read_csv(input_filepath, encoding_errors='ignore')
    print(f'Opened {input_filepath}')

    # Merge target
    df = merge_target(df, input_filepath)

    # Determine significance
    insignificant_cols = []
    pass_list = ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU', 'TARGET']

    for col in df.columns:
        if col not in pass_list and not is_significant(df, col):
            insignificant_cols.append(col)
            print(col, 'is insignificant and will be dropped!')

    # Write to CSV file
    df.drop(insignificant_cols, axis=1).to_csv(output_filepath, index=False)
    print(f'Written to {output_filepath}')


if __name__ == "__main__":
    # Run arguments parser
    parser = argparse.ArgumentParser(description=
        """This script is purposed for feature selection from 
        application_train.csv or etc files based on stat tests. 
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
