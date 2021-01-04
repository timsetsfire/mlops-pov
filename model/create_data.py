import pandas as pd 
import os
import yaml
import json
# import pyodbc
import pandas as pd
import yaml
from pyathena import connect

def join_state_info(code_dir, data):
    ## simple dataframe read csv
    state_info = pd.read_csv(os.path.join(code_dir, "data/US_Zip_Code_Validation_Ranges.csv"))
    ## via athena
    # creds = yaml.load( open(os.path.join(code_dir,"athena_creds.yaml"), "rb"), Loader=yaml.FullLoader)
    # conn = connect(aws_access_key_id=creds["AWSAccessKeyId"],
    #              aws_secret_access_key=creds["AWSSecretKey"],
    #              s3_staging_dir="s3://cfds-athena-demo2/",
    #              region_name="ap-south-1")
    # state_info = pd.read_sql("SELECT * FROM cfds_athena_demo.us_zip_codes", conn)
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
    numeric_features = feature_type_dict["Numeric"]
    categorical_features = feature_type_dict["Categorical"]
    data = join_state_info(code_dir, data) 
    data = process_dates(code_dir, data)  
    data = clean_up(code_dir, data)  
    cols = data.columns 
    drop_these = list(set(cols).difference(set(numeric_features)).difference(set(categorical_features)))
    return data.drop(drop_these, axis=1)