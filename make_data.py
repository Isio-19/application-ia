from utils import script_error_print

import pandas as pd
import numpy as np
import sys
import time

def get_files_name():
    file_names = pd.read_csv("OUVRAGES.csv")["Ouvrage"].astype("str")

    return_array = []
    for file in file_names:
        if file[0] == "#":
            continue
        return_array.append(file)

    return return_array

def get_percentage_na(file_name):
    file = pd.read_csv(f"data/raw/{file_name}.csv")
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
        # comment if the file has too many NA's
        count_na = get_percentage_na(file)
        count_na = np.divide(count_na, 237)

        too_many_na_flag = False
        for var in count_na:
            if var > na_threshhold:
                too_many_na_flag = True
                break

        # comment the file if the validation and testing data NA% are over 40%
        if not too_many_na_flag: 
            file_content = pd.read_csv(f"data/raw/{file}.csv")
            
            index1 = len(file_content) -24
            index2 = len(file_content) -12
            
            # validation
            temp_df = file_content.loc[index1:index2-1, "GWL"]
            na_percent = temp_df.isna().sum()
            na_percent /= 12 
            if na_percent > 0.4:
                too_many_na_flag = True
            
            # testing
            temp_df = file_content.loc[index2:, "GWL"]
            na_percent = temp_df.isna().sum()
            na_percent /= 12 
            if na_percent > 0.4:
                too_many_na_flag = True
            
        if too_many_na_flag:
            index = files[files.Ouvrage == file].index[0]
            file_names[index] = "#"+str(file_names[index])

    files["Ouvrage"] = file_names

    files.to_csv("OUVRAGES.csv", index=False)

def offset_file(file_name):
    file_content = pd.read_csv(f"data/temp_files/{file_name}.csv")
    
    for offset in range(1, 7):
        column_name=f"GWL+{offset}"
        offsetted_values = file_content["GWL"].shift(-offset)
        file_content.insert(offset+1, column=column_name, value=offsetted_values)
        
    file_content.drop(file_content.tail(6).index, inplace=True)
    file_content.to_csv(f"data/temp_files/{file_name}.csv", index=False)

def normalize_list(list):
    mean = np.mean(list)
    std = np.std(list)
    return [(val-mean)/std for val in list]

def normalize_file(file_name):
    file_content = pd.read_csv(f"data/raw/{file_name}.csv")

    file_content["GWL"] =   normalize_list(file_content["GWL"])
    file_content["P"] =     normalize_list(file_content["P"])
    file_content["T"] =     normalize_list(file_content["T"])
    file_content["ET"] =    normalize_list(file_content["ET"])
    file_content["NDVI"] =  normalize_list(file_content["NDVI"])

    file_content.to_csv(f"data/temp_files/{file_name}.csv", index=False)

def mean_on_file(file_name):
    file_path = f"data/raw/{file_name}.csv"
    if NORMALIZE:
        file_path = f"data/temp_files/{file_name}.csv"
    
    file_content = pd.read_csv(file_path)

    for var in ["GWL", "P", "T" ,"ET", "NDVI"]:
        file_content["GWL"] =  file_content["GWL"].fillna(var)

    file_content.to_csv(f"data/temp_files/{file_name}.csv", index=False)

def mean_on_month(file_name):
    file_path = f"data/raw/{file_name}.csv"
    if NORMALIZE:
        file_path = f"data/temp_files/{file_name}.csv"
    
    file_content = pd.read_csv(file_path)
    
    # backup, incase all the values for a certain month are all NA
    means = {
        "GWL": 0,
        "P": 0,
        "T": 0,
        "ET": 0,
        "NDVI": 0,
    }
    
    for var in means:
        means[var] = np.nanmean(file_content[var]) 
    
        month_mean = [0 for _ in range(12)]
        
        for month_index in range(12):
            month_str = str(month_index+1)
            if len(month_str) < 2:
                month_str = "0" + month_str
                
            month_mask = file_content["date"].str.contains(rf"[0-9]+-{month_str}-[0-9]+")
            
            selected_dates = file_content[month_mask]
            nan_mask = selected_dates[var].notna()
            values_to_mean = selected_dates[nan_mask]
        
            if len(values_to_mean) == 0:
                month_mean[month_index] = means[var]
            else:
                month_mean[month_index] = values_to_mean[var].sum() / len(values_to_mean)
            
            file_content.loc[month_mask, [var]] = file_content.loc[month_mask, [var]].fillna(month_mean[month_index]) 
        
    file_content.to_csv(f"data/temp_files/{file_name}.csv", index=False)

def mean_file(file):
    match MEAN_TYPE:
        case "file":
            mean_on_file(file)
        case "month":
            mean_on_month(file)
        case _:
            print(Exception("Invalid value for MEAN_TYPE"))
            
def split_file(file_name):
    file_content = pd.read_csv(f"data/temp_files/{file_name}.csv")

    first_index  = len(file_content) - 24
    second_index = len(file_content) - 12

    train_data = file_content[:first_index]
    dev_data   = file_content[first_index:second_index]
    test_data  = file_content[second_index:]

    train_data.to_csv(f"data/split_data/train_data/{file_name}.csv", index=False)
    dev_data.to_csv(f"data/split_data/dev_data/{file_name}.csv", index=False)
    test_data.to_csv(f"data/split_data/test_data/{file_name}.csv", index=False)

def main():
    if not QUIET:
        print("Commenting OUVRAGES.csv")
    comment_files(NA_THRESHHOLD)
    
    for file in get_files_name():
        if NORMALIZE:   
            normalize_file(file)

        if not QUIET:
            print("Filling the NanS with mean")
        mean_file(file)

        offset_file(file)

        if not QUIET:
            print("Spliting the data files")
        split_file(file)

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
    
main()