from utils import is_int, is_float, is_string, script_error_print

import pandas as pd
import numpy as np
import sys

def get_files_name():
    file_names = pd.read_csv("OUVRAGES.csv")["Ouvrage"].astype("str")

    return_array = []
    for file in file_names:
        if file[0] == "#":
            continue
        return_array.append(file)

    return return_array

def get_percentage_na(file_name):
    file = pd.read_csv(f"data/{file_name}.csv")
    gwl_na  = file["GWL"].isna().sum()
    p_na    = file["P"].isna().sum()
    t_na    = file["T"].isna().sum()
    et_na   = file["ET"].isna().sum()
    ndvi_na = file["NDVI"].isna().sum()

    return [gwl_na, p_na, t_na, et_na, ndvi_na]

def comment_files(na_threshhold):
    if na_threshhold == -1:
        na_threshhold = 0.2

    files = pd.read_csv("OUVRAGES.csv")

    # reset the file name field
    files["Ouvrage"] = [file.replace("#", "") for file in files["Ouvrage"].astype("str")]
    
    file_names = files["Ouvrage"].to_numpy()

    for file in file_names:
        count_na = get_percentage_na(file)
        count_na = np.divide(count_na, 237)

        too_many_na_flag = False
        for var in count_na:
            if var > na_threshhold:
                too_many_na_flag = True
                break

        if too_many_na_flag:
            index = files[files.Ouvrage == file].index[0]
            file_names[index] = "#"+str(file_names[index])

    files["Ouvrage"] = file_names

    files.to_csv("OUVRAGES.csv", index=False)

def normalize_list(list):
    mean = np.mean(list)
    std = np.std(list)
    return [(val-mean)/std for val in list]

def normalize_file(file_name):
    file_content = pd.read_csv(f"data/{file_name}.csv")

    file_content["GWL"] =   normalize_list(file_content["GWL"])
    file_content["P"] =     normalize_list(file_content["P"])
    file_content["T"] =     normalize_list(file_content["T"])
    file_content["ET"] =    normalize_list(file_content["ET"])
    file_content["NDVI"] =  normalize_list(file_content["NDVI"])

    file_content.to_csv(f"normalized_data/{file_name}.csv", index=False)

def normalize_all_files():
    files = get_files_name()
    for file in files:
        if file[0] == "#":
            continue
        normalize_file(file)

def mean_on_file(file_name):
    file_path = f"data/{file_name}.csv"
    if NORMALIZE:
        file_path = f"normalized_data/{file_name}.csv"
    
    file_content = pd.read_csv(file_path)
    GWL_mean  = np.nanmean(file_content["GWL"])
    file_content["GWL"] =  file_content["GWL"].fillna(GWL_mean)
    file_content.to_csv(f"filled_data/{file_name}.csv", index=False)

def mean_on_month(file_name):
    file_path = f"data/{file_name}.csv"
    if NORMALIZE:
        file_path = f"normalized_data/{file_name}.csv"
    
    file_content = pd.read_csv(file_path)
    
    # backup, incase all the values for a certain month are all NA
    GWL_mean  = np.nanmean(file_content["GWL"])
    
    month_mean = [0 for _ in range(12)]
    
    for month_index in range(12):
        month_str = str(month_index+1)
        if len(month_str) < 2:
            month_str = "0" + month_str
            
        month_mask = file_content["date"].str.contains(rf"[0-9]+-{month_str}-[0-9]+")
        
        selected_dates = file_content[month_mask]
        nan_mask = selected_dates["GWL"].notna()
        values_to_mean = selected_dates[nan_mask]
       
        if len(values_to_mean) == 0:
            month_mean[month_index] = GWL_mean
        else:
            month_mean[month_index] = values_to_mean["GWL"].sum() / len(values_to_mean)
        
        file_content.loc[month_mask, ["GWL"]] = file_content.loc[month_mask, ["GWL"]].fillna(month_mean[month_index]) 
        
    file_content.to_csv(f"filled_data/{file_name}.csv", index=False)

def mean_all_files():
    files = get_files_name()

    if MEAN_TYPE == "file":
        for file in files: 
            mean_on_file(file)
    if MEAN_TYPE == "month":
        for file in files:
            mean_on_month(file)

def split_file(file_name):
    file_content = pd.read_csv(f"filled_data/{file_name}.csv")

    first_index  = len(file_content) - 24
    second_index = len(file_content) - 12

    train_data = file_content[:first_index]
    dev_data   = file_content[first_index:second_index]
    test_data  = file_content[second_index:]

    train_data.to_csv(f"split_data/train_data/{file_name}.csv", index=False)
    dev_data.to_csv(f"split_data/dev_data/{file_name}.csv", index=False)
    test_data.to_csv(f"split_data/test_data/{file_name}.csv", index=False)

def split_all_files():
    files = get_files_name()

    for file in files: 
        split_file(file)


def main(na_threshhold):
    if not QUIET:
        print("Commenting OUVRAGES.csv")
    comment_files(na_threshhold)
    if NORMALIZE:   
        normalize_all_files()

    if not QUIET:
        print("Filling the NanS with mean")
    mean_all_files()

    if not QUIET:
        print("Spliting the data files")
    split_all_files()

# Detect only single date point NAN's (and averaging them)

# get args
NA_THRESHHOLD = -1
NORMALIZE = False
QUIET = False
MEAN_TYPE = "file"

args = iter(sys.argv)
for var in args:
    try:
        match var: 
            case "-na" | "--na_threshhold":
                var = next(args)
                if not var.replace(".", "", 1).isdigit():
                    raise Exception(f"Excepted a float after the -na arg, not {var}")
                NA_THRESHHOLD = float(var)
            case "-n" | "--normalize":
                NORMALIZE = True
            case "-q" | "--quiet":
                QUIET = True
            case "-m" | "--mean":
                var = next(args)
                if not var in ["file", "month"]:
                    raise Exception(f"Expected 'file' or 'month' after -m arg, not {var}")
                
                MEAN_TYPE = var
            case "make_data.py":
                pass
            case _:
                raise Exception(f"Unexpected argument: {var}")
    except Exception as e:
        print(e)
        script_error_print("make_data.py")
        exit()
    
main(NA_THRESHHOLD)
