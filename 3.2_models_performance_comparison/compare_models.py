# Basiс libs
import argparse
import joblib
import numpy as np
import pandas as pd

# Models
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def logreg_gs(df: pd.DataFrame) -> GridSearchCV:
    """
    Receives a raw dataframe as input. Runs the pipeline with separate
    processing for numeric and categorical features. Places features in
    the GridSearchCV. Returns trained LogReg model as output.

    Parameters:
        :param df : The entire raw DataFrame.
    Returns:
        res : GridSearchCV class instance.
    """
    # X-train, y-target
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']

    # Num features pipeline
    num_features = X.select_dtypes(exclude='object').columns.to_list()
    num_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
        ])
    # Cat features pipeline
    cat_features = X.select_dtypes(include='object').columns.to_list()
    cat_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='XNA')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
        ])
    # Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features),
        ])
    # Prediction pipeline w/ LogisticRegression
    estimator = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])
    # Parameters
    param_grid = dict(
        classifier__max_iter=[30],
        classifier__penalty=['l1', 'l2'],
        classifier__solver=['saga', 'sag'],
        classifier__C=[0.1, 1, 10],
        classifier__random_state=[42],
    )
    # Grid search
    logreg = GridSearchCV(estimator, param_grid, cv=3,
                          scoring='roc_auc', n_jobs=-1, verbose=10).fit(X, y)

    return logreg


def catboost_gs(df: pd.DataFrame) -> CatBoostClassifier:
    """
    Receives a raw dataframe as input. Runs CatBoost Classifier with
    native Grid Search engine. Returns trained CatBoost model as output.

    Parameters:
        :param df : The entire raw DataFrame.
    Returns:
        res : CatBoostClassifier class instance.
    """
    # X-train, y-target
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']

    # Cat features list
    cat_features = X.select_dtypes(include='object').columns.to_list()

    # CatBoost Classifier
    catboost = CatBoostClassifier(
        eval_metric='AUC:hints=skip_train~false',
        silent=True,
        cat_features=cat_features,
    )
    # Parameters
    param_grid = dict(
        learning_rate=[0.1, 0.5],
        depth=[5, 7],
        iterations=[300, 400],
        auto_class_weights=['Balanced'],
        random_state=[42],
    )
    # Grid search
    catboost.grid_search(
        param_grid, X, y, cv=3,
        refit=True, shuffle=True,
        stratified=True, verbose=False
    )

    return catboost


def main(input_filepath: str, output_filepath: str) -> None:
    """
    Runs a grid search for Logistic Regression and Catboost models.
    Measures models quality using ROC-AUC metric. Saves the best
    models to files. Outputs results.
    
    Parameters:
        :param input_filepath : The filepath to input csv file.
        :param output_filepath : Where to save the best estimators.
    Returns:
        res : Prints out ROC-AUC metrics for the best estimators.
    """
    # Read CSV file
    df = pd.read_csv(input_filepath, encoding_errors='ignore', index_col='SK_ID_CURR')
    print(f'Opened {input_filepath}')

    # Features selected using stat. tests
    to_drop = [
    'AMT_ANNUITY',
    'FLAG_MOBIL',
    'WEEKDAY_APPR_PROCESS_START',
    'NONLIVINGAPARTMENTS_AVG',
    'NONLIVINGAPARTMENTS_MODE',
    'NONLIVINGAPARTMENTS_MEDI',
    'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_12',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_WEEK']

    df=df.drop(to_drop, axis=1)

    # Run & save LogReg model as Pickle
    print('\nLogisticRegression :')
    logreg = logreg_gs(df)
    joblib.dump(logreg.best_estimator_, output_filepath + 'logreg_gs.pkl', compress=1)

    # Run & save CatBoost model as CatBoost binary
    print('\nCatBoostClassifier :')
    catboost = catboost_gs(df)
    catboost.save_model(output_filepath + 'catboost_gs.cbm', format="cbm")

    # Output LogisticRegression results
    print('\nLogisticRegression :')
    print('\n\t Best ROC-AUC : %.5f' % logreg.best_score_)
    print('\n\t Best parameters :\n\t', logreg.best_params_)

    # Output CatBoostClassifier results
    print('\nCatBoostClassifier :')
    print('\n\t Best ROC-AUC : %.5f' % catboost.get_best_score()['learn']['AUC'])
    print('\n\t Best parameters :\n\t', catboost.get_params())


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
        Output path and filename, I.e. / <- save in current folder
        """)
    args = parser.parse_args()
    
    # Wrap paths in easy-to-read variables
    input_filepath = args.i
    output_filepath = args.o

    # Run main
    main(input_filepath, output_filepath)
