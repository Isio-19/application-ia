import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def get_files_name():
    content = pd.read_csv("OUVRAGES.csv").astype("str")

    names = []
    lat = []
    long = []
    region = []
    for index in range(len(content)):
        if content["Ouvrage"][index][0] == "#":
            continue

        names.append(content["Ouvrage"][index])
        lat.append(content["Latitude"][index])
        long.append(content["Longitude"][index])
        region.append(content["Region"][index])

    return names, lat, long, region

def get_percentage_na(file_name):
    file = pd.read_csv(f"data/raw/{file_name}.csv")
    gwl_na  = file["GWL"].isna().sum()
    p_na    = file["P"].isna().sum()
    t_na    = file["T"].isna().sum()
    et_na   = file["ET"].isna().sum()
    ndvi_na = file["NDVI"].isna().sum()

    return [gwl_na, p_na, t_na, et_na, ndvi_na]

def comment_files():
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
            if var > NA_THRESHOLD:
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
            if na_percent > NA_THRESHOLD:
                too_many_na_flag = True
            
            # testing
            temp_df = file_content.loc[index2:, "GWL"]
            na_percent = temp_df.isna().sum()
            na_percent /= 12 
            if na_percent > NA_THRESHOLD:
                too_many_na_flag = True
            
        if too_many_na_flag:
            index = files[files.Ouvrage == file].index[0]
            file_names[index] = "#"+str(file_names[index])

    files["Ouvrage"] = file_names

    files.to_csv("OUVRAGES.csv", index=False)

def offset_file(file_name):
    file_content = pd.read_csv(file_name)
    
    for offset in range(1, 7):
        column_name=f"GWL+{offset}"
        if column_name in file_content: 
            return
        
        offsetted_values = file_content["GWL"].shift(-offset)
        file_content.insert(offset+6, column=column_name, value=offsetted_values)
    
    file_content.drop(file_content.tail(6).index, inplace=True)

    if not os.path.exists("data/temp_files"):
        os.makedirs("data/temp_files")

    file_content.to_csv(file_name, index=False)

def normalize_list(list):
    mean = np.mean(list)
    std = np.std(list)
    return [(val-mean)/std for val in list]

def mean_on_file(file_name, file_path):
    file_content = pd.read_csv(file_path)

    for var in ["GWL", "P", "T" ,"ET", "NDVI"]:
        file_content["GWL"] =  file_content["GWL"].fillna(var)

    if not os.path.exists("data/temp_files"):
        os.makedirs("data/temp_files")

    file_content.to_csv(f"data/temp_files/{file_name}.csv", index=False)

def mean_on_month(file_name, file_path):   
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
        
    if not os.path.exists("data/temp_files"):
        os.makedirs("data/temp_files")
    
    file_content.to_csv(f"data/temp_files/{file_name}.csv", index=False)

def replace_null(file_name, file_path):
    file_content = pd.read_csv(file_path)
    
    for var in ["GWL", "P", "T" ,"ET", "NDVI"]:
        file_content[var] = file_content[var].fillna(-1)

    if not os.path.exists("data/temp_files"):
        os.makedirs("data/temp_files")
    
    file_content.to_csv(f"data/temp_files/{file_name}.csv", index=False)
    
def mean_file(file):
    path = f"data/raw/{file}.csv"
    if NORMALIZE == "all":
        path = f"data/norm/{file}.csv"
    elif NORMALIZE == "no_gwl": 
        path = f"data/norm_no_gwl/{file}.csv"

    if (MEAN_TYPE == "file"): mean_on_file(file, path)
    elif (MEAN_TYPE == "month"): mean_on_month(file, path)
    else: replace_null(file, path)

def split_date(file):
    file_content = pd.read_csv(file)
          
    if ("year" in file_content) or ("month" in file_content) :
        return
          
    split = [i.split("-") for i in file_content["date"].to_numpy()]
    year = [i[0] for i in split]
    month = [i[1] for i in split]
            
    file_content.insert(4, column="year", value=year)
    file_content.insert(5, column="month", value=month)

    file_content.to_csv(file, index=False)
          
def split_file(file_name):
    file_content = pd.read_csv(f"data/temp_files/{file_name}.csv")

    first_index  = len(file_content) - 24
    second_index = len(file_content) - 12

    train_data = file_content[:first_index]
    dev_data   = file_content[first_index:second_index]
    test_data  = file_content[second_index:]

    for path in ["train_data", "dev_data", "test_data"]:
        if not os.path.exists(f"data/split_data/{path}"):
            os.makedirs(f"data/split_data/{path}")

    train_data.to_csv(f"data/split_data/train_data/{file_name}.csv", index=False)
    dev_data.to_csv(f"data/split_data/dev_data/{file_name}.csv", index=False)
    test_data.to_csv(f"data/split_data/test_data/{file_name}.csv", index=False)

#  PART 1
def create_norm_and_no_gwl_files(file):
    file_content = pd.read_csv(f"data/raw/{file}.csv")

    for var in ["P", "T", "ET", "NDVI"]:
        file_content[var] = normalize_list(file_content[var])
    file_content.to_csv(f"data/norm_no_gwl/{file}.csv", index=False)

    file_content["GWL"] = normalize_list(file_content[var])
    file_content.to_csv(f"data/norm/{file}.csv", index=False)

def add_position(file, lat, long, region):
    file_content = pd.read_csv(f"data/raw/{file}.csv")
    
    if not "Latitude" in file_content:
        file_content.insert(1, "Latitude", lat)
    if not "Longitude" in file_content:
        file_content.insert(1, "Longitude", long)
    if not "Region" in file_content:
        file_content.insert(1, "Region", region)
    
    file_content.to_csv(f"data/raw/{file}.csv", index=False)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    # Positional arguments
    parser.add_argument("na_threshold", type=float, help="set the threshhold of NANs under which the files are keep")
    parser.add_argument("mean_type", type=str, choices=["file", "month", "none"], help="which method is used the fill the NANs")

    # Optional arguments
    parser.add_argument("-n", "--normalize", type=str, choices=["all", "no_gwl"], help="normalize the files that are kept")
    parser.add_argument("-mf", "--make_file", action="store_true", help="generates the norm and norm_no_gwl files and make the appropriate transformations")
    parser.add_argument("-q", "--quiet", action="store_true",  help="suppresses the prints")

    args = parser.parse_args()

    if args.make_file:
        NA_THRESHOLD = 1
        comment_files()
        
        files, lat, long, region = get_files_name()
        for i, file in enumerate(files) :
            add_position(file, lat[i], long[i], region[i])

            if not os.path.exists("data/norm"):
                os.makedirs("data/norm")
            if not os.path.exists("data/norm_no_gwl"):
                os.makedirs("data/norm_no_gwl")

            create_norm_and_no_gwl_files(file)
            
            for type in ["raw", "norm", "norm_no_gwl"]:
                path = f"data/{type}/{file}.csv"
                split_date(path)
                offset_file(path)
            
    NA_THRESHOLD = float(args.na_threshold)
    MEAN_TYPE = args.mean_type
    NORMALIZE = args.normalize
    QUIET = args.quiet

    if not QUIET:
        print("Commenting OUVRAGES.csv")
    comment_files()

    if not QUIET:
        print("Processing the files")

    files, _, _, _ = get_files_name()
    for file in files:
        mean_file(file)
        split_file(file)
        
    if not QUIET: 
        print("Finished make data")
