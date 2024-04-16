import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def normalize(array, norm: bool = False):
    if not norm:
        return array

    mean = np.mean(array)
    std = np.std(array)

    return [(val-mean)/std for val in array]

def ccc(first_data, second_data):
    covar = np.cov(first_data, second_data)
    first_mean = np.mean(first_data)
    second_mean = np.mean(second_data)
    first_std = np.std(first_data)
    second_std = np.std(second_data)

    return (2 * covar)/( (first_mean - second_mean)**2 + first_std**2 + second_std**2 )
 
def read_data():
    return

def make_graph(number: int = 5, normalized: bool = False):
    files = pd.read_csv("OUVRAGES.csv")

    # first number files
    files_to_open = files["Ouvrage"][:number]
    if number == 0:
        files_to_open = files["Ouvrage"]

    for csv_file in files_to_open:
        file = pd.read_csv("Data/"+str(csv_file)+".csv")

        x = file["date"]
        y_gwl   = normalize(file["GWL"], normalized)
        y_p     = normalize(file["P"], normalized)
        y_t     = normalize(file["T"], normalized)
        y_et    = normalize(file["ET"], normalized)
        y_ndvi  = normalize(file["NDVI"], normalized)

        gwl_line,   = plt.plot(x, y_gwl, label="GWL")
        p_line,     = plt.plot(x, y_p, label="P")
        t_line,     = plt.plot(x, y_t, label="T")
        et_line,    = plt.plot(x, y_et, label="ET")
        ndvi_line,  = plt.plot(x, y_ndvi, label="NDVI")

        plt.legend(handles=[gwl_line, p_line, t_line, et_line, ndvi_line])
        plt.xticks([])

        if not normalized:
            plt.savefig("img/standard/"+str(csv_file)+".png", dpi=150)
        else:
            plt.savefig("img/normalized/"+str(csv_file)+".png", dpi=150)
        
        plt.clf()

make_graph(5, normalized=False)
make_graph(5, normalized=True)