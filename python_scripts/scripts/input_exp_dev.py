import yaml
import argparse
from datetime import datetime
import os, yaml, sys

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import get_experiment_parameters, update_experiments_log

parms = get_experiment_parameters()
update_experiments_log(parms['analyses_name'])
