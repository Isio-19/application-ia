import pandas as pd
import numpy as np
import math
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

def comment_files(na_threshhold = 0.2):
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

def mean_na(file_name):
    file_content = pd.read_csv(f"normalized_data/{file_name}.csv")
    GWL_mean  = np.nanmean(file_content["GWL"])
    P_mean    = np.nanmean(file_content["P"])
    T_mean    = np.nanmean(file_content["T"])
    ET_mean   = np.nanmean(file_content["ET"])
    NDVI_mean = np.nanmean(file_content["NDVI"])

    file_content["GWL"] =  file_content["GWL"].fillna(GWL_mean)
    file_content["P"] =    file_content["P"].fillna(P_mean)
    file_content["T"] =    file_content["T"].fillna(T_mean)
    file_content["ET"] =   file_content["ET"].fillna(ET_mean)
    file_content["NDVI"] = file_content["NDVI"].fillna(NDVI_mean)

    file_content.to_csv(f"filled_data/{file_name}.csv", index=False)

def mean_all_files():
    files = get_files_name()

    for file in files:
        mean_na(file)

def split_file(file_name, filled_data = False):
    path = "normalized_data"
    if filled_data:
        path = "filled_data"

    file_content = pd.read_csv(f"{path}/{file_name}.csv")

    first_index  = len(file_content) - 24
    second_index = len(file_content) - 12

    train_data = file_content[:first_index]
    dev_data   = file_content[first_index:second_index]
    test_data  = file_content[second_index:]

    train_data.to_csv(f"split_data/train_data/{file_name}.csv", index=False)
    dev_data.to_csv(f"split_data/dev_data/{file_name}.csv", index=False)
    test_data.to_csv(f"split_data/test_data/{file_name}.csv", index=False)

def split_all_files(fill_data):
    files = get_files_name()

    for file in files: 
        split_file(file, fill_data)

"""
Usage: python3 make_data.py -fill na 0.1
"""
def main(na_threshhold, fill_data):
    print("Commenting OUVRAGES.csv")
    comment_files(na_threshhold)
    normalize_all_files()

    if fill_data:
        print("Filling the NanS with mean")
        mean_all_files()

    print("Spliting the data files")
    split_all_files(fill_data)

# Detect only single date point NAN's (and averaging them)

# get args
na_threshhold = -1
fill_data = False

args = sys.argv
for i, var in enumerate(args):
    if var == "na" and (len(args) > i+1) and (args[i+1].replace(".", "", 1).isdigit()):
        na_threshhold = float(args[i+1])

    if var == "-fill":
        fill_data = True

# main(na_threshhold, fill_data)
