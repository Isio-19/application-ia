import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Apprendre un modèle sur un corpus d'apprentissage et de validation
Retirer la dernière année pour faire un corpus de test
Metrique : MSE

Modèle générale > Fine-tune sur chaque puis
"""

def normalize(array):
    mean = np.mean(array)
    std = np.std(array)

    return [(val-mean)/std for val in array]

def ccc(first_data, second_data):
    covar = np.cov(first_data, second_data)[0,1]
    first_mean = np.mean(first_data)
    second_mean = np.mean(second_data)
    first_var = np.var(first_data)
    second_var = np.var(second_data)

    return (2 * covar)/( (first_mean - second_mean)**2 + first_var + second_var )

def calculate_ccc(data):
    data = data.dropna(subset=["GWL"])
    gwl = data["GWL"]
    for other_column in ["P", "T", "ET", "NDVI"]:
        other_column_data = data[other_column]
        value = ccc(gwl, other_column_data)
        print(f"GWL-{other_column} : {value}")

def read_data():
    print("Reading data files")
    files = pd.read_csv("OUVRAGES.csv")

    files_to_open = files["Ouvrage"]

    file_names = []
    data_std = []
    data_norm = []
    for csv_file in files_to_open:
        file_names.append(csv_file)

        fileData_std = pd.read_csv("Data/standard/"+str(csv_file)+".csv")
        fileData_norm = pd.read_csv("Data/normalized/"+str(csv_file)+".csv")

        data_std.append(fileData_std)
        data_norm.append(fileData_norm)

    print("Finished reading data files")
    
    return file_names, data_std, data_norm

def normalize_data(data):
    return_data = []

    for d in data:
        insert_data = d.copy(deep=True)
        insert_data["GWL"]   = normalize(d["GWL"])
        insert_data["P"]     = normalize(d["P"])
        insert_data["T"]     = normalize(d["T"])
        insert_data["ET"]    = normalize(d["ET"])
        insert_data["NDVI"]  = normalize(d["NDVI"])
        return_data.append(insert_data)

    return return_data

def make_graph(file_name: str, data):
    x = data["date"]
    y_gwl   = data["GWL"]
    y_p     = data["P"]
    y_t     = data["T"]
    y_et    = data["ET"]
    y_ndvi  = data["NDVI"]

    gwl_line,   = plt.plot(x, y_gwl, label="GWL")
    p_line,     = plt.plot(x, y_p, label="P")
    t_line,     = plt.plot(x, y_t, label="T")
    et_line,    = plt.plot(x, y_et, label="ET")
    ndvi_line,  = plt.plot(x, y_ndvi, label="NDVI")

    plt.legend(handles=[gwl_line, p_line, t_line, et_line, ndvi_line])
    plt.xticks([])
    plt.savefig(f"img/{file_name}.png", dpi=150)
    plt.clf()


# read the files and normalize them
# file_names, std_data, _ = read_data()

# standard_file_names = [f"standard/{file_name}" for file_name in file_names]
# normalized_file_names = [f"normalized/{file_name}" for file_name in file_names]
# norm_data = normalize_data(std_data)

# for i, data in enumerate(norm_data):
#     data.to_csv(f"Data/{normalized_file_names[i]}.csv", index=False)

# make the graphs
# file_names, std_data, norm_data = read_data()
# standard_file_names = [f"standard/{file_name}" for file_name in file_names]
# normalized_file_names = [f"normalized/{file_name}" for file_name in file_names]

# for i, data in enumerate(std_data):
#     make_graph(standard_file_names[i], data)

# for i, data in enumerate(norm_data):
#     make_graph(normalized_file_names[i], data)

# calculate the ccc
# file_names, std_data, norm_data = read_data()
# for s_d, n_d in zip(std_data, norm_data):
#     print("CCC for standard data")
#     calculate_ccc(s_d)
#     print("CCC for normalized data")
#     calculate_ccc(n_d)