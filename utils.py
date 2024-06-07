import pandas as pd
import numpy as np

def get_files_name(region = None):
    file_content = pd.read_csv("OUVRAGES.csv")
    file_names = file_content["Ouvrage"].astype("str")
    file_region = file_content["Region"].astype("str")

    return_array = []
    for i, file in enumerate(file_names):
        if file[0] == "#":
            continue

        if region != None:
            if file_region[i] != region:
                continue
            
        return_array.append(file)

    return return_array
