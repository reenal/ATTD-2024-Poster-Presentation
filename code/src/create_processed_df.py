"""
The following script combines the files created in `./surv/data/interim` folder.
The final processed file is then used for creating survival model
NOTE:
    1. `merged_df`: Combined Dataframe of interim dataframes. It contains records for MaskID's with 5 or more records and for age over 1 year
    2. `survival_df`: Final Dataframe containing 1 Record/MaskID. This will be utilized for creating train-test dataset for Modeling.
"""
# os
import sys
sys.path.append("../")
from pathlib import Path
from config import Config

# numeric
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity

from tqdm import tqdm

# Utilities
from feature_engineering_utils import EstimateTimeWindowProbabilityDensity,DrawAgeProbability, EstimateAntibodySignalProbability




raw = Config.raw
assert raw.exists(), f"Check the folder påath containing raw files, current path = {raw}"
interim = Config.interim
assert interim.exists(), f"Check the folder påath containing raw files, current path = {interim}"
processed = Config.processed
assert processed.exists(), f"Check the folder påath containing raw files, current path = {processed}"

# Collect interim datasets

df_test = pd.read_feather(interim/"test_results_interim")
df_diagnosis = pd.read_feather(interim/"df_diagnosed")
df_physical = pd.read_feather(interim/"physical_exam")
## Create flag for T1D-positive patients
df_test["t1d"] = 0
df_test.loc[df_test.MaskID.isin(df_diagnosis.MaskID.unique()), "t1d"] =1

# Merge datasets

df_test = df_test.merge(df_diagnosis, on=["MaskID"], how="left", validate="m:1")
df_test = df_test.merge(df_physical[["MaskID", "DRAW_AGE", 'LENGTHHEIGHT','WEIGHT','SCREENING_FATHER_T1D', 'SCREENING_MOTHER_T1D',
       'SCREENING_SIBLING_T1D']], on=["MaskID", "DRAW_AGE"], how="right", validate="1:1")

## Required to merge static (Does not change with time) columns from physical-attributes related dataset
df_physical_static_cols = ["MaskID", 'RACE_ASIAN', 'RACE_BLACKORAFRICANAMERICAN',
       'RACE_NATIVEAMERICANALASKANNATI', 'RACE_NATIVEHAWAIIANOROTHERPACI',
       'RACE_UNKNOWNORNOTREPORTED', 'RACE_WHITE', 'BABYBIRTHTYPE_Singleton',
       'BABYBIRTHTYPE_Triplet', 'BABYBIRTHTYPE_Twin', 'MOMFIRSTCHILD_No',
       'MOMFIRSTCHILD_Yes', 'MOMFIRSTCHILD_unknown', 'SEX_Female', 'SEX_Male',
       ]
df_physical_static = df_physical.filter(items=df_physical_static_cols).drop_duplicates(subset=df_physical_static_cols, keep="first", ignore_index=True)

## We only include MaskID's that are common in 'Test results' and 'Physical attributes' related datasets
common_ids = np.intersect1d(df_test.MaskID.unique(), df_physical.MaskID.unique())
## Merge Test results dataset with static physical attributes value
df_test = df_test.merge(df_physical_static, on=["MaskID"], how="left", validate="m:1")

### Select MaskID's with 5 or more records per MaskID
more_than_five_rec = df_test.MaskID.value_counts()[df_test.MaskID.value_counts()>=5].index.unique()
#### Create conditions to further filter the dataset
cond_five_or_more = df_test.MaskID.isin(more_than_five_rec) # 5 or more records per maskID
cond_draw_age_filter = df_test.DRAW_AGE.between(365, 5160) # Records that have `DRAW_AGE` between 365 days (1 yr) and 5160 days (14 yr)
filtered_df = df_test[ cond_five_or_more & cond_draw_age_filter ] # Select rows based on conditions

# Save the combined dataframe as feather file in ´./surv/data/processed´ folder
filtered_df.reset_index().to_feather(processed/"merged_df")
print(f"Dataframe uploaded to path {processed}/merged_df")

# Feature Engineering

## Estimating Pr(t_{previous visit}<=t<= t_{current visit}) using Kernel Density Estimation
model_base = KernelDensity(bandwidth=0.1778279410038923, kernel="gaussian") # `bandwidth` estimated using 5-fold CV
model_base.fit(filtered_df.DRAW_AGE.astype("float64").values.reshape(-1,1))


ep = EstimateTimeWindowProbabilityDensity(model=model_base)

print("Estimating Time Window Probability Density...")
for window in tqdm(ep.one_day_time_window):
    ep.estimate_prob_density(values=window)

draw_age_prob = DrawAgeProbability(prob_dict=ep.prob_dict) # Object to estimate Draw Age Probability
estimate_ab = EstimateAntibodySignalProbability() # Object to estimate Probability of occurance of an autoantibody/MaskID/DRAW_AGE

## Create DataFrame with engineered features

df_for_survival = filtered_df.groupby("MaskID", group_keys=False)\
.apply(lambda grp: grp.assign(

    ## Features focusing on Multi AB positivity ## Check ´clean_test_results.py´ to understand how ´Recorded_AB_Pos´ and ´Multi_AB_Pos´ were created.
    MEDIAN_AB_POS = lambda df_: df_.Recorded_AB_Pos.median(),
    MEDIAN_MULTI_AB_POS_FLAG = lambda df_: df_.Multi_AB_Pos.median(),

    # NOTE: It is assumed that minimum "DRAW_AGE" > 365
    DRAW_AGE_HISTORY = lambda df_: [tuple(age) if idx!=0 else (365,int(age)) for idx, age in enumerate(df_.DRAW_AGE.rolling(1, closed="both"))],
    DRAW_AGE_DENSITY = lambda df_: df_.DRAW_AGE_HISTORY.apply(draw_age_prob.time_window_density),
    # Method-2 (Markov-chain)
    PROB_GAD_MEMORY   = lambda df_:  [estimate_ab.get_last_visit_prob(df_["GAD_PERSISTENCE_NEW"].values[:i+1])   for i in range(len(df_))] ,
    PROB_MIAA_MEMORY  = lambda df_:  [estimate_ab.get_last_visit_prob(df_["MIAA_PERSISTENCE_NEW"].values[:i+1])  for i in range(len(df_))] ,
    PROB_IA2A_MEMORY  = lambda df_:  [estimate_ab.get_last_visit_prob(df_["IA2A_PERSISTENCE_NEW"].values[:i+1])  for i in range(len(df_))] ,
    PROB_ZnT8A_MEMORY = lambda df_:  [estimate_ab.get_last_visit_prob(df_["ZnT8A_PERSISTENCE_NEW"].values[:i+1]) for i in range(len(df_))] ,
    PROB_MULTI_MEMORY = lambda df_:  [estimate_ab.get_last_visit_prob(df_["MULTI_PERSISTENCE_NEW"].values[:i+1]) for i in range(len(df_))] ,
    # Create ´BEFORE_DIAGNOSIS´, this feature specifies whether a record for a MaskID is taken before or after it was identified to be T1D positive
    BEFORE_DIAGNOSIS = lambda df_: np.where(((df_.DRAW_AGE < df_.DIAGNOSISDATEAGE) & (df_.t1d==1)), 1,0)
)).groupby(by=["MaskID"], group_keys=False)\
.apply(
    lambda grp: grp.assign(
        SCORE_GAD_MEMORY   = lambda df_:  np.cumsum(( df_.GAD_PERSISTENCE_NEW.values *    df_.PROB_GAD_MEMORY    * df_.DRAW_AGE_DENSITY))  ,
        SCORE_MIAA_MEMORY  = lambda df_: np.cumsum( (  df_.MIAA_PERSISTENCE_NEW.values *   df_.PROB_MIAA_MEMORY  * df_.DRAW_AGE_DENSITY)),
        SCORE_IA2A_MEMORY  = lambda df_: np.cumsum( (  df_.IA2A_PERSISTENCE_NEW.values *   df_.PROB_IA2A_MEMORY  * df_.DRAW_AGE_DENSITY)),
        SCORE_ZnT8A_MEMORY = lambda df_: np.cumsum( (  df_.ZnT8A_PERSISTENCE_NEW.values *  df_.PROB_ZnT8A_MEMORY * df_.DRAW_AGE_DENSITY)),
        SCORE_MULTI_MEMORY = lambda df_: np.cumsum( (  df_.MULTI_PERSISTENCE_NEW.values *  df_.PROB_MULTI_MEMORY * df_.DRAW_AGE_DENSITY))
    )
)

print("Successfully created df")

### These are columns we are interested for Modeling purposes
selected_columns = ["MaskID","DRAW_AGE", "SAMPLE_COLLECTION_VISIT", 
                    # Auto-Antibody related columns
                    "GAD_Pos", "GAD_unknown", "IA2A_Pos", "IA2A_unknown", "MIAA_Pos", "MIAA_unknown", "ZnT8A_Pos","ZnT8A_unknown", 
                    # Auto-Antibody Persistence related columns
                    "GAD_PERSISTENCE_NEW", "MIAA_PERSISTENCE_NEW", "IA2A_PERSISTENCE_NEW", "ZnT8A_PERSISTENCE_NEW", "MULTI_PERSISTENCE_NEW",
                    # Target Columns
                    "t1d", "DIAGNOSISDATEAGE", 
                    # Static Columns
                    "LENGTHHEIGHT", "WEIGHT", "SCREENING_FATHER_T1D", "SCREENING_MOTHER_T1D", "SCREENING_SIBLING_T1D", 
                    "RACE_ASIAN", "RACE_BLACKORAFRICANAMERICAN", "RACE_NATIVEAMERICANALASKANNATI", "RACE_NATIVEHAWAIIANOROTHERPACI", "RACE_UNKNOWNORNOTREPORTED", "RACE_WHITE",
                    "BABYBIRTHTYPE_Singleton", "MOMFIRSTCHILD_Yes", "MOMFIRSTCHILD_unknown", "SEX_Female", 
                    # Dynamic (Temporally) Columns
                    "MEDIAN_AB_POS", "MEDIAN_MULTI_AB_POS_FLAG",
                    "SCORE_GAD_MEMORY", "SCORE_MIAA_MEMORY", "SCORE_IA2A_MEMORY", "SCORE_ZnT8A_MEMORY", "SCORE_MULTI_MEMORY",
                    # Helper Columns
                    "DRAW_AGE_HISTORY", "DRAW_AGE_DENSITY", "BEFORE_DIAGNOSIS", 
                    ]


# Select Record's over the age of 1 year.
df_for_survival = df_for_survival.filter(items=selected_columns).query("DRAW_AGE>=365")

# Fill NA's for T1D-negative patients with latest draw age. This is done to ensure `DIAGNOSISDATEAGE` column has values for all the MaskID's (T1D-pos as well as T1D-neg)
df_for_survival["DIAGNOSISDATEAGE"] = df_for_survival["DIAGNOSISDATEAGE"].fillna(df_for_survival["DRAW_AGE"])

# Create Sub datasets for T1D-positive and T1D-negative
## NOTE: `df_t1d` captures last record/MaskID after which (i.e., in the next visit) the MaskID is classified to be T1D
df_t1d = df_for_survival[(df_for_survival.t1d==1) & (df_for_survival.BEFORE_DIAGNOSIS==1)].groupby("MaskID", group_keys=False).last() 
df_non_t1d = df_for_survival[df_for_survival.t1d==0].groupby("MaskID", group_keys=False).last()


survival_df = pd.concat([df_t1d, df_non_t1d], axis=0).reset_index()

survival_df.to_feather(processed/"survival_df")

print(f"Dataset for creating Survival Models is created in the `{processed}/survival_df` file.")