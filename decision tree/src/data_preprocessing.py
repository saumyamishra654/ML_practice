import pandas as pd
import numpy as np

def preprocess(filepath):
    dataset = pd.read_csv(filepath)

    # unnecessary columns related to flagged fraud, origin and destination names, and destination balances
    dataset = dataset.drop(columns=['isFlaggedFraud', 'nameOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'])

    # create new column 'full_withdraw' to indicate if 'amount' matches 'oldbalanceOrg' (1 if true, 0 if false)
    dataset['full_withdraw'] = dataset.apply(lambda row: 1 if row['amount'] == row['oldbalanceOrg'] else 0, axis=1)

    #  create a column to flag datapoints in the suspicious range 55-110. 10-> high risk, 5-> low risk
    dataset['suspicious_time_frame'] = dataset['step'].apply(lambda x: 10 if 55 <= x <= 110 else 5)

    # binary fraud indicator: 1 for 'CASH_OUT' and 'TRANSFER', 0 for others
    dataset['type'] = dataset['type'].apply(lambda x: 1 if x in ['CASH_OUT', 'TRANSFER'] else 0)

    # drop columns no longer needed: 'oldbalanceOrg', 'newbalanceOrig', and 'step'
    dataset = dataset.drop(columns=['oldbalanceOrg', 'newbalanceOrig', 'step'])

    # shuffle data
    dataset = dataset.sample(frac=1, random_state=50).reset_index(drop=True)

    return dataset

