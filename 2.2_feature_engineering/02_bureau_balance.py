import argparse
import pandas as pd


def compile_bureau(df_bureau: pd.DataFrame, df_bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features based on manual preset.

    Parameters:
        :param df_bureau : DataFrame with bureau.csv.
        :param df_bureau_balance : DataFrame with bureau_balance.csv.
    Returns:
        res : Returns a pivot table with a column for each status,
        as well as the interval before opening and closing for each loan.
    """
    # Initializing DataFrame with SK_ID_BUREAU as index, and SK_ID_CURR mapping with Application DataFrame
    df = pd.DataFrame(index=df_bureau_balance.SK_ID_BUREAU.unique())
    df = df.join(df_bureau.set_index('SK_ID_BUREAU')['SK_ID_CURR'], how='inner')

    # Iterates over the STATUS flags and creates columns of the same name
    sta_flags = ['C', 'X', '0', '1', '2', '3', '4', '5']
    for flag in sta_flags:
        df[flag] = df_bureau_balance[df_bureau_balance.STATUS==flag].groupby('SK_ID_BUREAU').last()['STATUS']
    
    # 1 - STATUS present, 0 - STATUS flag absent | for unique SK_ID_BUREAU (one ID = one loan)
    df = df.replace({'C':1, 'X':1, '0':1, '1':1, '2':1, '3':1, '4':1, '5':1})
    df[sta_flags] = df[sta_flags].fillna(0).astype('int')

    # A = Active loan; MONTH_CLOSED - when first C appears; MONTH_OPENED – when first record appears
    df['A'] = abs(df['C'] - 1)
    df['MONTH_CLOSED'] = df_bureau_balance[df_bureau_balance.STATUS=='C'].groupby('SK_ID_BUREAU').last()['MONTHS_BALANCE']
    df['MONTH_OPENED'] = df_bureau_balance[df_bureau_balance.STATUS!='C'].groupby('SK_ID_BUREAU').last()['MONTHS_BALANCE']

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features based on manual preset.

    Parameters:
        :param df : Initial DataFrame.
    Returns:
        res : DataFrame with generated features.
    """
    df_final = pd.DataFrame()

    # 1. ACTIVE_CREDIT_CNT: Number of open loans
    df_final['ACTIVE_CREDIT_CNT'] = df.groupby('SK_ID_CURR')['A'].sum()

    # 2. CLOSED_CREDIT_CNT: Number of closed loans
    df_final['CLOSED_CREDIT_CNT'] = df.groupby('SK_ID_CURR')['C'].sum()

    # 3. DPD_.._CNT: Number of overdue loans for different days of delay (DPD-1,.. DPD-5)
    df_final['DPD_1_CNT'] = df.groupby('SK_ID_CURR')['1'].sum()
    df_final['DPD_2_CNT'] = df.groupby('SK_ID_CURR')['2'].sum()
    df_final['DPD_3_CNT'] = df.groupby('SK_ID_CURR')['3'].sum()
    df_final['DPD_4_CNT'] = df.groupby('SK_ID_CURR')['4'].sum()
    df_final['DPD_5_CNT'] = df.groupby('SK_ID_CURR')['5'].sum()

    # 4. CREDITS_CNT: Number of loans
    df_final['CREDITS_CNT'] = df_final['ACTIVE_CREDIT_CNT'] + df_final['CLOSED_CREDIT_CNT']

    # 5. CLOSED_CREDITS_RATE: Share of closed loans
    df_final['CLOSED_CREDITS_RATE'] = df_final['CLOSED_CREDIT_CNT'] / df_final['CREDITS_CNT']

    # 6. ACTIVE_LOANS_RATE: Share of open loans
    df_final['ACTIVE_CREDITS_RATE'] = 1 - df_final['CLOSED_CREDITS_RATE']

    # 7. DPD_.._RATE: The share of overdue loans for different days of delay (DPD-1,.. DPD-5)
    df_final['DPD_CNT'] = df_final.loc[:, 'DPD_1_CNT':'DPD_5_CNT'].sum(axis=1)
    df_final['DPD_1_RATE'] = (df_final['DPD_1_CNT'] / df_final['DPD_CNT']).fillna(0)
    df_final['DPD_2_RATE'] = (df_final['DPD_2_CNT'] / df_final['DPD_CNT']).fillna(0)
    df_final['DPD_3_RATE'] = (df_final['DPD_3_CNT'] / df_final['DPD_CNT']).fillna(0)
    df_final['DPD_4_RATE'] = (df_final['DPD_4_CNT'] / df_final['DPD_CNT']).fillna(0)
    df_final['DPD_5_RATE'] = (df_final['DPD_5_CNT'] / df_final['DPD_CNT']).fillna(0)

    # 8. The interval between the last closed loan and the current application
    df_final['LAST_CLOSED_INTERVAL'] = df.groupby('SK_ID_CURR')['MONTH_CLOSED'].max()
    # WARNING: empty values are filled with zeroes.
    df_final['LAST_CLOSED_INTERVAL'] = df_final['LAST_CLOSED_INTERVAL'].fillna(0)

    # 9. The interval between taking the last active loan and the current application
    df_final['LAST_ACTIVE_OPENED_INTERVAL'] = df[df['A'] == 1].groupby('SK_ID_CURR')['MONTH_OPENED'].max()
    # WARNING: empty values are filled with zeroes.
    df_final['LAST_ACTIVE_OPENED_INTERVAL'] = df_final['LAST_ACTIVE_OPENED_INTERVAL'].fillna(0)

    return df_final


def main(bureau_filepath: str, balance_filepath: str, output_filepath: str) -> None:
    """
    Generates features from bureau_balance.csv and outputs to csv file.

    Parameters:
        :param bureau_filepath : The filepath to bureau.csv file.
        :param balance_filepath : The filepath to bureau_balance.csv file.
        :param output_filepath : The filepath to output csv file.
    Returns:
        res : Outputs the file with the generated features.
    """ 
    # Read CSV files
    df_bureau = pd.read_csv(bureau_filepath, encoding_errors='ignore')
    print(f'Opened {bureau_filepath}')

    df_bureau_balance = pd.read_csv(balance_filepath, encoding_errors='ignore')
    print(f'Opened {balance_filepath}')

    # Gererate new features
    df = compile_bureau(df_bureau, df_bureau_balance)
    df = feature_engineering(df)

    # Write to CSV file
    df.to_csv(output_filepath, index=True)  # SK_ID_CURR is used as an index col
    print(f'Written to {output_filepath}')

    
if __name__ == "__main__":
    # Run arguments parser
    parser = argparse.ArgumentParser(description=
        """This script is purposed for feature generation from 
        bureau_balance.csv file. 
        | Source: Home Credit Default Risk Kaggle competition 
        (https://www.kaggle.com/c/home-credit-default-risk/data)
        """)
    parser.add_argument('--bur', type=str, help=
        """
        Path and filename, I.e. /bureau.csv
        """)
    parser.add_argument('--bal', type=str, help=
        """
        Path and filename, I.e. /bureau_balance.csv
        """)
    parser.add_argument('-o', type=str, help=
        """
        Output path and filename, I.e. /bureau_balance.csv
        """)
    args = parser.parse_args()
    
    # Wrap paths in easy-to-read variables
    bureau_filepath = args.bur
    balance_filepath = args.bal
    output_filepath = args.o

    # Run main
    main(bureau_filepath, balance_filepath, output_filepath)
