import argparse
from dataclasses import (asdict, dataclass)
import json
from typing import TextIO
from tqdm import tqdm


@dataclass  # for JSON type=='bureau'
class AmtCredit:
    CREDIT_CURRENCY: str
    AMT_CREDIT_MAX_OVERDUE: float
    AMT_CREDIT_SUM: float
    AMT_CREDIT_SUM_DEBT: float
    AMT_CREDIT_SUM_LIMIT: float
    AMT_CREDIT_SUM_OVERDUE: float
    AMT_ANNUITY: float


@dataclass  # for JSON type=='POS_CASH_balance'
class PosCashBalanceIDs:
    SK_ID_PREV: int
    SK_ID_CURR: int
    NAME_CONTRACT_STATUS: str


def create_output_files(bureau_filepath: str, pos_filepath: str) -> None:
    '''
    Receives paths to bureau.csv and POS_CASH_balance.csv files;
    Creates these two files with the given headers.
    '''
    # the header for bureau.csv
    bureau_header = [
        'SK_ID_CURR', 'SK_ID_BUREAU', 'CREDIT_ACTIVE', 'CREDIT_CURRENCY',
        'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
        'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
        'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
        'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT_UPDATE',
        'AMT_ANNUITY'
    ]   
    # writes the given header to POS_CASH_balance.csv
    with open(bureau_filepath, 'w') as file: 
        file.write(','.join(bureau_header)+'\n')    
   
    # the header for bureau.csv
    pos_header = [
        'SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT',
        'CNT_INSTALMENT_FUTURE', 'NAME_CONTRACT_STATUS', 'SK_DPD',
        'SK_DPD_DEF'
    ]
    # writes the given header to POS_CASH_balance.csv
    with open(pos_filepath, 'w') as file: 
        file.write(','.join(pos_header).replace('None', '')+'\n')


def bureau(data: dict, bureau_file: TextIO, line_number: int) -> None:
    '''
    Receives a JSON-object with type=='bureau' and line number as input;
    Transforms object to string, writes to bureau.csv;
    '''
    try:
        # using dataclass AmtCredit to transform class to dict
        data['record']['AmtCredit'] = asdict(eval(data['record']['AmtCredit']))
        # transforms JSON to string
        data_list = [
            str(data['record']['SK_ID_CURR']),
            str(data['record']['SK_ID_BUREAU']),
            str(data['record']['CREDIT_ACTIVE']),
            str(data['record']['AmtCredit']['CREDIT_CURRENCY']),        
            str(data['record']['DAYS_CREDIT']),
            str(data['record']['CREDIT_DAY_OVERDUE']),            
            str(data['record']['DAYS_CREDIT_ENDDATE']),            
            str(data['record']['DAYS_ENDDATE_FACT']),
            str(data['record']['AmtCredit']['AMT_CREDIT_MAX_OVERDUE']),
            str(data['record']['CNT_CREDIT_PROLONG']),   
            str(data['record']['AmtCredit']['AMT_CREDIT_SUM']),
            str(data['record']['AmtCredit']['AMT_CREDIT_SUM_DEBT']),            
            str(data['record']['AmtCredit']['AMT_CREDIT_SUM_LIMIT']),
            str(data['record']['AmtCredit']['AMT_CREDIT_SUM_OVERDUE']),
            str(data['CREDIT_TYPE']),              
            str(data['record']['DAYS_CREDIT_UPDATE']),
            str(data['record']['AmtCredit']['AMT_ANNUITY']),
        ]
        # writes to file
        # add replace('None', '') for 100% identity
        bureau_file.write(','.join(data_list)+'\n')
    except:
        print(f"Error parsing {line_number}: {data}")


def pos_cash_balance(data: dict, pos_file: TextIO, line_number: int) -> None:
    '''
    Receives a JSON-object with type=='POS_CASH_balance' and line number as input;
    Transforms object to string, writes to POS_CASH_balance.csv;

    '''
    for entry in data['records']:
        try:
            # using dataclass PosCashBalanceIDs to transform class to dict
            entry['PosCashBalanceIDs'] = asdict(eval(entry['PosCashBalanceIDs']))
            # transforms JSON to string
            one_row = [
                str(entry['PosCashBalanceIDs']['SK_ID_PREV']),
                str(entry['PosCashBalanceIDs']['SK_ID_CURR']),
                str(entry['MONTHS_BALANCE']),
                str(data['CNT_INSTALMENT']),
                str(entry['CNT_INSTALMENT_FUTURE']),
                str(entry['PosCashBalanceIDs']['NAME_CONTRACT_STATUS']),
                str(entry['SK_DPD']),
                str(entry['SK_DPD_DEF']),
            ]
            # writes to file
            # add replace('None', '') for 100% identity
            pos_file.write(','.join(one_row)+'\n')
        except:
            print(f"Error parsing {line_number}: {entry}")


def func_switch(json: dict, bureau_file: TextIO, pos_file: TextIO, line_number: int) -> None:
    '''
    Receives a JSON object, output files, and line number.
    Determines which function the date will be handled.
    '''
    if json['type'] == 'bureau':
        bureau(json['data'], bureau_file, line_number)
    else:
        pos_cash_balance(json['data'], pos_file, line_number)


def main(input_path: str, bur_output_path: str, pos_output_path: str) -> None:
    # set file names
    input_filepath = str(input_path or '') + 'POS_CASH_balance_plus_bureau.log'
    bureau_filepath = str(bur_output_path or '') + 'bureau.csv'
    pos_filepath = str(pos_output_path or '') + 'POS_CASH_balance.csv'

    # create output files
    create_output_files(bureau_filepath, pos_filepath)

    # read input file
    input_file = open(input_filepath, 'r')

    # open output files for appending
    bureau_file = open(bureau_filepath, 'a')
    pos_file = open(pos_filepath, 'a')

    # parsing input file line by line
    for line_number, raw_line in tqdm(enumerate(input_file)):
        try:
            func_switch(json.loads(raw_line), bureau_file, pos_file, line_number)
        except:
            print(f'An error occurred while parsing a raw line {line_number}')

    # closing files
    input_file.close()
    bureau_file.close()
    pos_file.close()


if __name__ == "__main__":
    # run arguments parser
    parser = argparse.ArgumentParser(
        description='''Script for parsing POS_CASH_balance_plus_bureau.log.
        Outputs bureau.csv and POS_CASH_balance.csv files.''')
    parser.add_argument('-i', type=str, help='Input dir with POS_CASH_balance_plus_bureau.log')
    parser.add_argument('-b', type=str, help='Output dir for bureau.csv')
    parser.add_argument('-p', type=str, help='Output dir for POS_CASH_balance.csv')
    args = parser.parse_args()

    input_path = args.i
    bur_output_path = args.b
    pos_output_path = args.p

    # run main with argparse arguments
    main(input_path, bur_output_path, pos_output_path)
