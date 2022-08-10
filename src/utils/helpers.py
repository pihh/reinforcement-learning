import os
import json
import time 
import hashlib
import pandas as pd 

def parse_time(seconds):
    time_format = time.strftime("%H:%M:%S", time.gmtime(seconds))
    return time_format

def parse_date(df):
    parsed_date = pd.to_datetime(df.date).dt.date
    df['date'] = parsed_date

    return parsed_date
def get_number_of_files_in_folder(directory):
    return len(os.listdir(directory))

def md5(json_obj):
    return hashlib.md5(json.dumps(json_obj,sort_keys=True, indent=2).encode('utf-8')).hexdigest()