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

# Folder paths
raw = Config.raw
interim = Config.interim

assert (interim/"test_results_interim").exists(), f"The file has not been created. To create run file 'src/clean_test_results.py'"

if (interim/"test_results_interim").exists():
    df_test_results = pd.read_feather(interim/"test_results_interim")
    only_ids = df_test_results.drop_duplicates(subset = ["MaskID"],keep="first").MaskID.unique() # MaskID for participants who continued after screening

screening_form = pd.read_csv(raw/"SCREENING_FORM.csv").query("MaskID.isin(@only_ids)")

# Specify features to be selected
imp_cols = ["ANYFAMILYMEMT1D", "BABYBIRTHTYPE", "MOMFIRSTCHILD", "SEX","FDRPREVIOUSVALUE",
            'RACE_ASIAN','RACE_BLACKORAFRICANAMERICAN','RACE_NATIVEAMERICANALASKANNATI','RACE_NATIVEHAWAIIANOROTHERPACI','RACE_UNKNOWNORNOTREPORTED','RACE_WHITE', 
            "WHICHFAMILYMEMT1D_FATHER", "WHICHFAMILYMEMT1D_MOTHER", "WHICHFAMILYMEMT1D_SIBLING",
            "FDR", "EVENT_AGE", "HLADRAWAGE", "MaskID"]

screening_form = screening_form.filter(items=imp_cols)

# Data-cleaning
screening_form[imp_cols[0]] = screening_form[imp_cols[0]].str.replace("Unknown", "unknown")
screening_form[imp_cols[4]] = screening_form[imp_cols[4]].str.replace("Unknown", "unknown")
screening_form[imp_cols[1:5]] = screening_form[imp_cols[1:5]].fillna(value="unknown")
screening_form[imp_cols[5:14]] = screening_form[imp_cols[5:14]].fillna(value=0)
screening_form["FDR"] = screening_form["FDR"].fillna(-1)
screening_form = screening_form.dropna(how="any", subset=["EVENT_AGE", "HLADRAWAGE", "MaskID"])

## specify data-types
int_dtype_dict = {col: Config.int_dtype for col in imp_cols[5:]}
screening_form = screening_form.astype(int_dtype_dict)\
.rename(columns={"ANYFAMILYMEMT1D": "SCREENING_ANYFAMILYMEMT1D",
                 "FDR": "SCREENING_FDR",
                 "WHICHFAMILYMEMT1D_FATHER": "SCREENING_FATHER_T1D",
                 "WHICHFAMILYMEMT1D_MOTHER": "SCREENING_MOTHER_T1D",
                 "WHICHFAMILYMEMT1D_SIBLING": "SCREENING_SIBLING_T1D"
                 })

# Save data
screening_form.reset_index(drop=True).to_feather(interim/"screening_form")