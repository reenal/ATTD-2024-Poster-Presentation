# os
import sys
sys.path.append("../")
from pathlib import Path

import pandas as pd
from config import Config

raw = Config.raw

assert raw.exists(), f"Check the folder påath containing raw files, current path = {raw}"
# D-types
int_dtype = Config.int_dtype
float_dtype = Config.float_dtype

df_diagnosis = pd.read_csv(raw/"DIABETES_DIAGNOSIS.csv")

df_diagnosed = df_diagnosis.loc[~df_diagnosis.MaskID.isna(),["MaskID", "DIAGNOSISDATEAGE"]].astype(
    {
        "MaskID" : int_dtype,
        "DIAGNOSISDATEAGE": int_dtype
    }
)

df_diagnosed.reset_index(drop=True).to_feather(Config.interim/"df_diagnosed")

