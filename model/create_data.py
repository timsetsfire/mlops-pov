import pandas as pd 
import os
import yaml
import json


def join_state_info(code_dir, data):
    state_info = pd.read_csv(os.path.join(code_dir, "data/US_Zip_Code_Validation_Ranges.csv"))
    df = data.merge(state_info, how="left", left_on = ["zip_code", "addr_state"], right_on = ["zip", "addr_state"])
    return df

def process_dates(code_dir, data):
    with open(os.path.join(code_dir, "feature_detail.yaml"), "r") as f:
        feature_type_dict = yaml.load(f, Loader=yaml.FullLoader)
    date_fields = feature_type_dict["Date"]
    for date_field in date_fields:
        data[date_field] = pd.to_datetime(data[date_field])
        data["{} day of week".format(date_field)] = data[date_field].dt.weekday
        data["{} month of year".format(date_field)] = data[date_field].dt.month
    return data

def clean_up(code_dir, data):
    data["int_rate"] = data["int_rate"].apply(lambda x: x.replace("%", "")).astype(float)
    return data

def write_schema(code_dir, data):
    schema = data.dtypes.to_json()
    schema = [(k,v["name"]) for k,v in json.loads(schema).items()]
    with open(os.path.join(code_dir, "schema.json"), "w") as f:
        d = json.dumps(dict(schema))
        f.write(d)

def process_data(code_dir, data):
    with open(os.path.join(code_dir, "feature_detail.yaml"), "r") as f:
        feature_type_dict = yaml.load(f, Loader=yaml.FullLoader)
    data = join_state_info(code_dir, data) 
    data = process_dates(code_dir, data)  
    data = clean_up(code_dir, data)  
    return data