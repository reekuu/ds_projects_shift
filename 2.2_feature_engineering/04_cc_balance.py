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
    # 1. Count all possible aggregates on the cards
    values = [
        'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
        'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
        'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
        'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
        'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
        'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF'
    ]
    aggfunc = ['mean', 'std', 'var', 'first', 'last', 'min', 'max']

    # All possible aggregates
    df_pivot = df.pivot_table(index='SK_ID_CURR', values=values, aggfunc=aggfunc)

    # 2. Calculate how aggregates have changed over the last 3 months
    df_pivot_3m = df[df.MONTHS_BALANCE >= -3].pivot_table(index='SK_ID_CURR', values=values, aggfunc=aggfunc)
    aggregates_change_over3m = df_pivot_3m / df_pivot

    # 3. Final DF
    # WARNING: empty values are filled with zeroes.
    df_final = pd.concat([df_pivot, aggregates_change_over3m], axis=1).fillna(0)

    # 4. Rename multi-index columns names
    cols_short_names = []
    for col in df_final.columns:
        short_name = '_'.join(col[::-1])
        cols_short_names.append(short_name)
    
    df_final.columns = cols_short_names

    return df_final


def main(input_filepath: str, output_filepath: str) -> None:
    """
    Generates features from credit_card_balance.csv and outputs to csv file.

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
        credit_card_balance.csv file. 
        | Source: Home Credit Default Risk Kaggle competition 
        (https://www.kaggle.com/c/home-credit-default-risk/data)
        """)
    parser.add_argument('-i', type=str, help=
        """
        Path and filename, I.e. /credit_card_balance.csv
        """)
    parser.add_argument('-o', type=str, help=
        """
        Output path and filename, I.e. /credit_card_balance.csv
        """)
    args = parser.parse_args()
    
    # Wrap paths in easy-to-read variables
    input_filepath = args.i
    output_filepath = args.o

    # Run main
    main(input_filepath, output_filepath)
    
