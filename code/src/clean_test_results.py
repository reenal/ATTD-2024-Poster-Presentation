# os
import sys
sys.path.append("../")
from pathlib import Path

# numeric
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 70)
import matplotlib.pyplot as plt
from config import Config


raw = Config.raw

assert raw.exists(), f"Check the folder påath containing raw files, current path = {raw}"
# D-types
int_dtype = Config.int_dtype
float_dtype = Config.float_dtype

# Load Data
df_test_result = pd.read_csv(raw/"TEST_RESULTS.csv")

# Antibody tests
ab_tests = ["GAD", "IA2A", "MIAA", "ZnT8A"]
# 
outcomes = ["Neg", "Pos"]
assay_ids = [9932,9933]
# DRAW_AGE cannot be greater than EVALUATE_AGE or RECEIVE_AGE
# RESULT, DRAW_AGE,OUTCOME cannot be NA
# OUTCOME with only 'Neg' and 'Pos' values are cosidered
# Separate datasets for separate ASSAY IDs (NOTE: For ASSAY_ID==10596.0, the results are not reported)!
df_test_result = df_test_result.query("""TEST_NAME.isin(@ab_tests) & ~((DRAW_AGE>RECEIVE_AGE) | (DRAW_AGE>EVALUATE_AGE)) & ~RESULT.isna() & ~DRAW_AGE.isna() & ~OUTCOME.isna() & OUTCOME.isin(@outcomes)""")\
.filter(items = ["TEST_NAME", "RESULT", "OUTCOME", "ANTIBODY_SPECNAME", "SAMPLE_COLLECTION_VISIT","DRAW_AGE", "MaskID", "ASSAY_ID"])\
.astype({"TEST_NAME":"object",
         "RESULT": float_dtype,
         "OUTCOME": "object",
          "SAMPLE_COLLECTION_VISIT": int_dtype,
          "DRAW_AGE": int_dtype,
          "MaskID": int_dtype,
          "ASSAY_ID": int_dtype})\
          .sort_values(by=["MaskID","DRAW_AGE","TEST_NAME"])\
            .reset_index()

            # NOTE: To be done later
          #.assign(OUTCOME = lambda df_: df_.OUTCOME.replace("Not Reported (changed by Dev)", "unknown"))

### Dropped rows when there is a consent
df_test_result = df_test_result.drop_duplicates(subset = ["MaskID","DRAW_AGE","OUTCOME", "TEST_NAME"])

# Dataframe containing only unique IDs wrt each 'SAMPLE_COLLECTION_VISIT'
only_ids = df_test_result.drop_duplicates(subset = ["MaskID", "DRAW_AGE"],keep="first").filter(items=["MaskID", "DRAW_AGE", "SAMPLE_COLLECTION_VISIT"])

correct_merged_df = pd.DataFrame(data=None) # Dataframe to store correctly noted AB test-results
not_merged_df = pd.DataFrame(data=None) # Dataframe to store records per MaskID/per DRAW_AGE where consensus on results could not be obtained§

grps = df_test_result.query("TEST_NAME== @ab_tests").groupby(by=["MaskID","DRAW_AGE"])
len_grps = len(grps)

# Iterate over groups    
for i, grp in enumerate(grps):
    df = grp[1]
    
    try:
        # To create the columns recording the outcome of AB Tests
        df1 = df.pivot(index=["MaskID","DRAW_AGE"], columns="TEST_NAME", values="OUTCOME")\
        .reset_index()
        correct_merged_df = pd.concat([correct_merged_df, df1], axis=0)
    # To handle records with ambiguity i.e., having more than one outcomes per MaskID/per DRAW_AGE
    except ValueError:
        # If only one outcome is obtained from different labs
        if df.OUTCOME.nunique()==1:
            df2 = df.head(1).pivot(index=["MaskID","DRAW_AGE"], columns="TEST_NAME", values="OUTCOME")\
        .reset_index()

        # If multiple outcomes are obtained and one of them is "Pos"
        elif (df.OUTCOME.nunique()>1) and ("Pos" in df.OUTCOME.unique()):
            cond_positive = df.OUTCOME == "Pos"
            df2 = df[cond_positive].head(1).pivot(index=["MaskID","DRAW_AGE"], columns="TEST_NAME", values="OUTCOME")\
        .reset_index()

        # If multiple outcomes are obtained and none of them is "Pos"
        elif (df.OUTCOME.nunique()>1) and not ("Pos" in df.OUTCOME.unique()):
            df2 = df.head(1).pivot(index=["MaskID","DRAW_AGE"], columns="TEST_NAME", values="OUTCOME")\
        .reset_index()
        # Record ambigous cases
        else:
            df2 = pd.DataFrame(data=None)
            not_merged_df = pd.concat([not_merged_df, df], axis=0)
        correct_merged_df = pd.concat([correct_merged_df, df2], axis=0)

correct_merged_df.reset_index().to_feather(Config.interim/"correct_merged_df")

not_merged_df.reset_index().to_feather(Config.interim/"not_merged_df")

correct_merged_df = pd.read_feather(Config.interim/"correct_merged_df").fillna("unknown")
assert len(correct_merged_df) <= len(grps), "Merged dataframes cannot be greater!"

ab_dataset = only_ids.merge(correct_merged_df, on=["MaskID", "DRAW_AGE"], how="left", validate="1:1").fillna(value="unknown")

## Feature Engineering

def diff_draw_age_flag(x):
        
        """Function to calculate difference in 'titers_drawdt_agedys', grouped over 'MaskID'.

        Args:
            x (_type_): _description_

        Returns:
            int: 0 for False else 1 for True
        """
        diff =  x.iloc[-1] - x.iloc[0]

        if np.isnan(diff) or (diff<=21):
            return 0
        else:
            return 1

## Dummify the AB Tests
## Create "Recorded_AB_Pos" to record how many of tests were positive per record/per MaskID/per DRAW_AGE
## Create "Multi_AB_Pos" to flag if more than one AB tests were positive per record/per MaskID/per DRAW_AGE
ab_dummy = pd.get_dummies(ab_dataset, columns=ab_tests, dummy_na=False, dtype=int_dtype)

test_results_interim = ab_dummy.assign(
              Recorded_AB_Pos = lambda row: (row[['GAD_Pos', 'IA2A_Pos', 'MIAA_Pos', 'ZnT8A_Pos']].sum(axis=1)),
              Multi_AB_Pos = lambda df_: np.where(df_["Recorded_AB_Pos"]>1,1,0)
              ).astype({"Recorded_AB_Pos": int_dtype, "Multi_AB_Pos": int_dtype})\
              .sort_values(by=["MaskID", "DRAW_AGE"])\
              .groupby(by=["MaskID"], group_keys=True)\
              .apply(lambda grp: grp.assign(
                  # AB-Persistence definitions defined in TEDDY Protocol
                  GAD_PERSISTENCE_OLD = lambda df_: np.where(df_.GAD_Pos.rolling(2).sum().shift(-1).fillna(value=0)==2, 1,0).astype(dtype=int_dtype),
                  MIAA_PERSISTENCE_OLD = lambda df_: np.where(df_.MIAA_Pos.rolling(2).sum().shift(-1).fillna(value=0)==2, 1,0).astype(dtype=int_dtype),
                  IA2A_PERSISTENCE_OLD = lambda df_: np.where(df_.IA2A_Pos.rolling(2).sum().shift(-1).fillna(value=0)==2, 1,0).astype(dtype=int_dtype),
                  ZnT8A_PERSISTENCE_OLD = lambda df_: np.where(df_.ZnT8A_Pos.rolling(2).sum().shift(-1).fillna(value=0)==2, 1,0).astype(dtype=int_dtype),
                  # Our Proposed definitions
                  GAD_PERSISTENCE_NEW = lambda df_: np.where((df_.GAD_Pos.rolling(2).sum().fillna(value=0)==2) & (df_.DRAW_AGE.rolling(2).apply(diff_draw_age_flag)), 1,0).astype(dtype=int_dtype),
                  MIAA_PERSISTENCE_NEW = lambda df_: np.where((df_.MIAA_Pos.rolling(2).sum().fillna(value=0)==2) & (df_.DRAW_AGE.rolling(2).apply(diff_draw_age_flag)), 1,0).astype(dtype=int_dtype),
                  IA2A_PERSISTENCE_NEW = lambda df_: np.where((df_.IA2A_Pos.rolling(2).sum().fillna(value=0)==2) & (df_.DRAW_AGE.rolling(2).apply(diff_draw_age_flag)), 1,0).astype(dtype=int_dtype),
                  ZnT8A_PERSISTENCE_NEW = lambda df_: np.where((df_.ZnT8A_Pos.rolling(2).sum().fillna(value=0)==2) & (df_.DRAW_AGE.rolling(2).apply(diff_draw_age_flag)), 1,0).astype(dtype=int_dtype),
              ))\
                .assign(
                    # Multi-persistence computed using old definition
                    MULTI_PERSISTENCE_OLD = lambda row: np.where(row[["GAD_PERSISTENCE_OLD", "MIAA_PERSISTENCE_OLD", "IA2A_PERSISTENCE_OLD", "ZnT8A_PERSISTENCE_OLD"]].sum(axis=1)>1, 1,0).astype(dtype=int_dtype),
                    # Multi-persistence computed using new definition
                    MULTI_PERSISTENCE_NEW = lambda row: np.where(row[["GAD_PERSISTENCE_NEW", "MIAA_PERSISTENCE_NEW", "IA2A_PERSISTENCE_NEW", "ZnT8A_PERSISTENCE_NEW"]].sum(axis=1)>1, 1,0).astype(dtype=int_dtype),
                    
                )



test_results_interim.droplevel(level=0).to_feather(Config.interim/"test_results_interim")