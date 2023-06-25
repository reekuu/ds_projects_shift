import argparse
import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features based on manual preset.

    Parameters:
        :param df : Initial DataFrame.
    Returns:
        res : DataFrame with generated features.
    """
    df_final = pd.DataFrame()

    # 1. CLIENT_MAX_OVERDUE: Client's maximum overdue sum
    df_final['CLIENT_MAX_OVERDUE'] = df.groupby(['SK_ID_CURR'])['AMT_CREDIT_MAX_OVERDUE'].max()

    # 2. CLIENT_MIN_OVERDUE: Client's minimum overdue sum
    df_final['CLIENT_MIN_OVERDUE'] = df.groupby(['SK_ID_CURR'])['AMT_CREDIT_MAX_OVERDUE'].min()

    # 3. CURR_OVERDUE_RATE: Which part of the sum is overdue
    overdue_rate = df[df.CREDIT_ACTIVE == 'Active']
    overdue_rate = overdue_rate.groupby('SK_ID_CURR')[['AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM']].sum()
    df_final['CURR_OVERDUE_RATE'] = overdue_rate['AMT_CREDIT_SUM_OVERDUE'] / overdue_rate['AMT_CREDIT_SUM']

    # 4. Total number of loans of a certain type
    total_loans_cnt = pd.pivot_table(
        df,
        index='SK_ID_CURR', columns='CREDIT_TYPE', values='DAYS_CREDIT_UPDATE',
        aggfunc='count', fill_value=0,
    ).add_prefix('CNT_')

    # 5. Number of overdue loans of a certain type
    overdue_loans_cnt = pd.pivot_table(
        df[df['CREDIT_DAY_OVERDUE'] > 0],
        index='SK_ID_CURR', columns='CREDIT_TYPE', values='CREDIT_DAY_OVERDUE',
        aggfunc='count', fill_value=0,
    ).add_prefix('DPD_')

    # 6. Number of closed loans of a certain type
    closed_loans_cnt = pd.pivot_table(
        df[df['CREDIT_ACTIVE'] == 'Closed'],
        index='SK_ID_CURR', columns='CREDIT_TYPE', values='CREDIT_ACTIVE',
        aggfunc='count', fill_value=0,
    ).add_prefix('CLOSED_CNT_')

    # Final DF
    select = [df_final, total_loans_cnt, overdue_loans_cnt, closed_loans_cnt]
    # WARNING: empty values are filled with zeroes.
    final_df = pd.concat(select, axis=1).fillna(0)

    return final_df


def main(input_filepath: str, output_filepath: str) -> None:
    """
    Generates features from bureau.csv and outputs to csv file.

    Parameters:
        :param input_filepath : The filepath to input csv file.
        :param output_filepath : The filepath to output csv file.
    Returns:
        res : Outputs the file with the generated features.
    """ 
    # Read CSV file
    df = pd.read_csv(input_filepath, encoding_errors='ignore')
    print(f'Opened {input_filepath}')

    # Gererate new features
    df = feature_engineering(df)
    
    # Write to CSV file
    df.to_csv(output_filepath, index=True)  # SK_ID_CURR is used as an index col
    print(f'Written to {output_filepath}')
    
    
if __name__ == "__main__":
    # Run arguments parser
    parser = argparse.ArgumentParser(description=
        """This script is purposed for feature generation from 
        bureau.csv file. 
        | Source: Home Credit Default Risk Kaggle competition 
        (https://www.kaggle.com/c/home-credit-default-risk/data)
        """)
    parser.add_argument('-i', type=str, help=
        """
        Path and filename, I.e. /bureau.csv
        """)
    parser.add_argument('-o', type=str, help=
        """
        Output path and filename, I.e. /bureau.csv
        """)
    args = parser.parse_args()
    
    # Wrap paths in easy-to-read variables
    input_filepath = args.i
    output_filepath = args.o

    # Run main
    main(input_filepath, output_filepath)
