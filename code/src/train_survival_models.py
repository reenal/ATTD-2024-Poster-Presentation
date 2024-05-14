"""
##################################################
## Script for training models. For generating results of Table-2
##################################################
# Author: Shahrukh Iqbal
# Email: iqbal@mainly.ai
##################################################
"""

# os
import sys
sys.path.append("../")
from pathlib import Path
import pickle
from config import Config
from argparse import ArgumentParser
# numeric
import numpy as np
import pandas as pd
from functools import reduce
# Modeling
from sklearn import set_config
set_config(display="text")  # displays text representation of estimators
from sklearn.model_selection import train_test_split

from sksurv.ensemble import (
    RandomSurvivalForest
    )
from sksurv.linear_model import (
    CoxPHSurvivalAnalysis
    )
from sksurv.tree import SurvivalTree

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from tqdm import tqdm


## Helper Functions start ##

def for_survival(y_old):
    data_y = y_old.to_numpy()
    aux_y = [(e1,e2) for e1,e2 in data_y]
    y_new = np.array(aux_y, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    return y_new


def data_within_training(df:pd.DataFrame, features,target):
    """Function to split data conditionally:
        1. Train Data Time window is greater than Test Data time window (Start and end of TEDDY trial)
        2. Train Data window includes 'DIAGNOSIS_AGE' lesser than 365 and greater than '5000' days 

    Args:
        df (pd.DataFrame): Pandas DataFrame for Survival analysis.
        features: X-covariates
        target: Y-covariates, in the order of (Indicator, Age)
    """
    df = df[features+target].drop_duplicates(subset=features, ignore_index=True)
    # Diagnosis age lesser than 730 days
    cond_one = (df[target[-1]]<=730)
    #cond_one = (df[target[-1]]<1)
    # Diagnosis age greater than 3650 days (10 years) and T1D+
    cond_two = (df[target[-1]]>4380) 
    #cond_two = (df[target[-1]]>3000) 

    # Condition for definitely selecting in trainig data
    cond_is_in_train = (cond_one)|(cond_two)

    # Divide into 2 DataFrames, 'df1' will be a part of Training
    # 'df2' time range is within 'df1'
    
    df1,df2 =  df[cond_is_in_train], df[~cond_is_in_train]
    return df1, df2

def create_train_test_split(df1:pd.DataFrame,df2:pd.DataFrame, features,target):
    """
    df1: First Dataframe created using `data_within_training` function
    df2: Second Dataframe created using `data_within_training` function
    """
    x_train,x_test, y_train, y_test = train_test_split(df2[features], 
                                                       df2[target],
                                                       test_size=Config.test_size, 
                                                       stratify=df2[target[0]], 
                                                       random_state=None, # None, required for Monte Carlo Cross Validation
                                                       shuffle=True)

    x_train = pd.concat([df1[features], x_train])
    y_train = pd.concat([df1[target], y_train])

    # Test for correct datasplit
    train_min, train_max = y_train[target[-1]].min(), y_train[target[-1]].max()
    test_min, test_max = y_test[target[-1]].min(), y_test[target[-1]].max()
    # assert train_min <= test_min < test_max <= train_max, "time range or test data is not within time range of training data."
    if (train_min <= test_min < test_max < train_max):
        return x_train, x_test, y_train,y_test
    else:
        return None

def standardise_y(y_train, y_test):
    """Required for feeding in scikit-survival"""
    # Transforming y into a format acceptable for scikit-survival
    y_train, y_test = list(map(for_survival, [y_train, y_test]))
    return y_train, y_test

def monte_carlo_cv(df1, df2, features, target, n_iterations=10):
    """
    This function has the following objectives:
        1. To create a Monte Carlo Split for the given dataset.
        2. To ensure that dataset is stratified and randomly shuffled each time it is split.
        3. Build survival models of type a). Cox-Proportion Hazard, b). Decision Trees and, c). Random Survival Forest
        4. The Train and Test dataset that are successfully created in each split and their respective models are saved in a python dictionary.
    """
    data_dict = {}
    for iteration in tqdm(range(n_iterations)):
        if create_train_test_split(df1,df2,features,target) is not None:
            # Train-Test Split
            x_train, x_test, y_train,y_test = create_train_test_split(df1,df2,features,target)
            y_train_std, _ = standardise_y(y_train, y_test)
            try:

                # Modeling
                # Fit simple PHCox
                cox_estimator = CoxPHSurvivalAnalysis()
                cox_estimator.fit(x_train,y_train_std)

                # # Decision Trees
                dt_estimator = SurvivalTree()
                dt_estimator.fit(x_train,y_train_std)

                # # Random Survival Forest
                rf_estimator = RandomSurvivalForest(n_estimators=50,
                                    min_samples_split=5,
                                    min_samples_leaf=3,
                                    n_jobs=-1,
                                    random_state=Config.model_random_state, # Only for replicability purpose
                                    warm_start = True,
                                    verbose=0)
                rf_estimator.fit(x_train,y_train_std)

                key = iteration
                value = {"x_train": x_train,
                         "x_test": x_test,
                         "y_train": y_train,
                         "y_test": y_test,
                         "cox_estimator": cox_estimator,
                         "dt_estimator": dt_estimator,
                         "rf_estimator": rf_estimator
                         }
                data_dict[key] = value
            except ValueError:
                continue
        else:
            continue
    return data_dict

## Helper Functions end ##

## Define Features and Target ##

features = ['GAD_Pos','GAD_unknown', 'IA2A_Pos', 'IA2A_unknown', 'MIAA_Pos', 'MIAA_unknown','ZnT8A_Pos', 'ZnT8A_unknown',
            'GAD_PERSISTENCE_NEW','MIAA_PERSISTENCE_NEW', 'IA2A_PERSISTENCE_NEW', 'ZnT8A_PERSISTENCE_NEW','MULTI_PERSISTENCE_NEW',
            'LENGTHHEIGHT','WEIGHT', 
            'SCREENING_FATHER_T1D', 'SCREENING_MOTHER_T1D','SCREENING_SIBLING_T1D', 
            'RACE_ASIAN', 'RACE_BLACKORAFRICANAMERICAN','RACE_NATIVEAMERICANALASKANNATI', 'RACE_NATIVEHAWAIIANOROTHERPACI','RACE_UNKNOWNORNOTREPORTED', 'RACE_WHITE', 
            'BABYBIRTHTYPE_Singleton',
            'MOMFIRSTCHILD_Yes', 'MOMFIRSTCHILD_unknown', 
            'SEX_Female',
            'MEDIAN_AB_POS', 'MEDIAN_MULTI_AB_POS_FLAG',
            'SCORE_GAD_MEMORY','SCORE_MIAA_MEMORY', 'SCORE_IA2A_MEMORY', 'SCORE_ZnT8A_MEMORY','SCORE_MULTI_MEMORY']

target = ['t1d', 'DIAGNOSISDATEAGE']


if __name__ == "__main__":

    processed = Config.processed
    assert processed.exists(), f"Check the folder path containing processed files, current path = {processed}"
    output = Config.output
    assert output.exists(), f"Check the folder path containing Output files, current path = {output}"
    survival_df = pd.read_feather(processed/"survival_df") # Load dataset from `.data/processed` folder

    parser = ArgumentParser()
    parser.add_argument("-n", "--niter", default=5,help="No. of Datasets and models to be created using Monte Carlo CV")
    args = parser.parse_args()

    df1, df2 = data_within_training(survival_df, features=features, target=target)

    data_dict = monte_carlo_cv(df1, df2, features, target, n_iterations=int(args.niter))

    with open(output/"MC_DATA_DICT.pickle", "wb") as file:
        pickle.dump(data_dict, file)
    file.close()
    print(f"File dumped in {output}")
