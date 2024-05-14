# os
import sys
sys.path.append("../")
from pathlib import Path

# numeric
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 70)
from config import Config

from sklearn.impute import KNNImputer

# Folder paths
raw = Config.raw
interim = Config.interim

# Data Loading
df_pe = pd.read_csv(raw/"PHYSICAL_EXAM.csv")
imp_cols = ["LENGTHHEIGHT","WEIGHT", "EVENT_AGE", "DATEMEASUREMENAGE", "DATEMEASUREMENTAGENONSTANDARD", "MaskID"]
# Drop rows if no AGE related info is availabe
df_pe = df_pe[imp_cols].dropna(how="all", subset=["EVENT_AGE", "DATEMEASUREMENAGE", "DATEMEASUREMENTAGENONSTANDARD"]).reset_index(drop=True)
# Assigns minium age of event out of 'EVENT_AGE', 'DATEMEASUREMENAGE', 'DATEMEASUREMENTAGENONSTANDARD'
df_pe["MIN_EVENT_AGE"] = df_pe.iloc[:,2:].min(axis=1).astype(Config.int_dtype)
df_pe = df_pe[["LENGTHHEIGHT","WEIGHT", "MIN_EVENT_AGE", "MaskID"]].sort_values(by=["MaskID", "MIN_EVENT_AGE"])

# Load Test-Results data
df_test_result = pd.read_csv(raw/"TEST_RESULTS.csv")

int_dtype = Config.int_dtype
float_dtype = Config.float_dtype
ab_tests = ["GAD", "IA2A", "MIAA", "ZnT8A"]
outcomes = ["Neg", "Pos"]
assay_ids = [9932,9933]
# DRAW_AGE cannot be greater than EVALUATE_AGE or RECEIVE_AGE
# RESULT, DRAW_AGE,OUTCOME cannot be NA
# OUTCOME with only 'Neg' and 'Pos' values are cosidered
# Separate datasets for separate ASSAY IDs (NOTE: For ASSAY_ID==10596.0, the results are not reported)!
df_test_result = df_test_result.query("""
                                      TEST_NAME.isin(@ab_tests) & ~((DRAW_AGE>RECEIVE_AGE) | (DRAW_AGE>EVALUATE_AGE)) & ~RESULT.isna() & ~DRAW_AGE.isna() & ~OUTCOME.isna() & OUTCOME.isin(@outcomes)
                                      """)\
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

            # NOTE: Ye baad me karnaa hai
          #.assign(OUTCOME = lambda df_: df_.OUTCOME.replace("Not Reported (changed by Dev)", "unknown"))

### Dropped rows when there is a consent i.e.,
### For a given MaskID/DRAW_AGE/TEST_NAME if the OUTCOME is same
df_test_result = df_test_result.drop_duplicates(subset = ["MaskID","DRAW_AGE","OUTCOME", "TEST_NAME"])
# Dataframe containing only unique IDs wrt each 'SAMPLE_COLLECTION_VISIT'
only_ids = df_test_result.drop_duplicates(subset = ["MaskID", "DRAW_AGE"],keep="first").filter(items=["MaskID", "DRAW_AGE", "SAMPLE_COLLECTION_VISIT"])

df_pe = df_pe.rename(columns={"MIN_EVENT_AGE": "DRAW_AGE"})\
.drop_duplicates(subset=["MaskID", "DRAW_AGE"], keep="first")

merge_with_maskids = only_ids.merge(df_pe, on=["MaskID", "DRAW_AGE"], how="left", validate="1:1")\
    .filter(items=['MaskID', 'DRAW_AGE', 'SAMPLE_COLLECTION_VISIT', 'LENGTHHEIGHT','WEIGHT'])

if (interim/"screening_form").exists():
    df_screening = pd.read_feather(interim/"screening_form")
    screening_vars = ["BABYBIRTHTYPE", "MOMFIRSTCHILD", "SEX", 
                  'RACE_ASIAN','RACE_BLACKORAFRICANAMERICAN','RACE_NATIVEAMERICANALASKANNATI','RACE_NATIVEHAWAIIANOROTHERPACI','RACE_UNKNOWNORNOTREPORTED','RACE_WHITE',
                  "MaskID"]
    merge_with_screening = merge_with_maskids.merge(df_screening.filter(items=screening_vars), on=["MaskID"], how="left", validate="m:1")\
.query("LENGTHHEIGHT<300")
    
# Create a dummy variable dataset
df_dummy = pd.get_dummies(data=merge_with_screening, drop_first=False, dtype=Config.int_dtype)

# Start Data-imputation
indicator =False
imputer = KNNImputer(missing_values=np.nan, n_neighbors=20, weights="distance", copy=True,add_indicator=indicator)

imputed_df = imputer.fit_transform(df_dummy[df_dummy.columns[1:].tolist()]) # Skip MaskIDs


if indicator:
    imputed_df = pd.DataFrame(data=imputed_df, columns=df_dummy.columns[1:].tolist()+ ["indicator"])
else:
    imputed_df = pd.DataFrame(data=imputed_df, columns=df_dummy.columns[1:].tolist())

final_df = pd.concat([df_dummy.MaskID.reset_index(drop=True), imputed_df], axis=1).astype(df_dummy[df_dummy.columns[1:].tolist()].dtypes)

# Add additional information about FRD-status
df_final = final_df.merge(df_screening.filter(items=["MaskID","SCREENING_FATHER_T1D", "SCREENING_MOTHER_T1D", "SCREENING_SIBLING_T1D"]), on=["MaskID"], how="left", validate="m:1")


df_final.to_feather(interim/"physical_exam")
