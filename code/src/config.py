import sys
#sys.path.append("../")
from pathlib import Path, PurePosixPath
from dataclasses import dataclass


@dataclass
class Config:
    # Paths
    data = Path("./data")
    interim = data/'interim'
    raw = data/'raw'
    processed = data/'processed'
    output = data/'outputs'

    # dtypes
    # PANDAS
    int_dtype = "int64"
    float_dtype = "float64"
    
    # Pandas config
    query_engine = 'python' # https://stackoverflow.com/questions/54759936/extension-dtypes-in-pandas-appear-to-have-a-bug-with-query
    # YAML
    config_yaml = Path("src/conf/config.yaml")

    # Model Paths
    model_path = Path("src/models")

    # Training Params
    split_random_state = 100
    test_size=0.2
    val_size=0.2
    model_random_state=10

