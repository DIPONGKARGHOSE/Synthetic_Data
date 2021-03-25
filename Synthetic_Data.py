# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 01:05:54 2021

@author: nijum
"""

import pandas as pd
from gretel_client import get_cloud_client

pd.set_option('max_colwidth', None)

client = get_cloud_client(prefix="api", api_key="grtucbfbc5356cc1d49757a29f3a4c7eff225b2a4bf907b052fab9acc3b9e1bfd5b4")
client.install_packages()
import pandas as pd

dataset_path = 'https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/healthcare-analytics-vidhya/train_data.csv'
nrows = 10000  # We will use this later when generating data
training_df = pd.read_csv(dataset_path, nrows=nrows)
training_df
from pathlib import Path

checkpoint_dir = str(Path.cwd() / "checkpoints")

config_template = {
    "checkpoint_dir": checkpoint_dir,
    "vocab_size": 20000,
    "overwrite": True
}
try:
    from gretel_helpers.synthetics import SyntheticDataBundle
except FileNotFoundError:
    from gretel_helpers.synthetics import SyntheticDataBundle



from gretel_helpers.synthetics import create_df, SyntheticDataBundle

model = SyntheticDataBundle(
    training_df=training_df,
    delimiter=None, # if ``None``, it will try and automatically be detected, otherwise you can set it
    auto_validate=True, # build record validators that learn per-column, these are used to ensure generated records have the same composition as the original
    synthetic_config=config_template, # the config for Synthetics
)
model.build()
model.train()
# num_lines: how many rows to generate
# max_invalid: the number of rows that do not pass semantic validation, if this number is exceeded, training will
# stop
model.generate(num_lines=nrows, max_invalid=nrows)
model.get_synthetic_df()
import IPython

report_path = './report.html'
model.generate_report(report_path=report_path)
IPython.display.HTML(filename=report_path)

model.save("my_model.tar.gz")

df = model.get_synthetic_df()
df.to_csv('synthetic-data.csv', index=False)

# Publish newly created synthetic data to a new private Gretel project 
project = client.get_project(display_name="Blueprint: Create Synthetic Data", create=True)
project.send_dataframe(df, detection_mode="all")
print(f"View this project at: {project.get_console_url()}")