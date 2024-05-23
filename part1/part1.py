import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_files_name():
    file_names = pd.read_csv("OUVRAGES.csv")["Ouvrage"].astype("str")

    return_array = []
    for file in file_names:
        if file[0] == "#":
            continue
        return_array.append(file)

    return return_array

def make_graph(file_content, path):
    x =  [i for i in range(len(file_content["date"]))]
    for var in ["P", "T", "ET", "NDVI", "GWL"]:
        plt.plot(x, file_content[var], label=f"{var}")
    plt.legend(loc="best")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.plot()
    plt.savefig(path)
    plt.clf()
    
def normalize_list(list):
    mean = np.mean(list)
    std = np.std(list)
    return [(val-mean)/std for val in list]

def normalize_file(file_content):
    file_content["GWL"] =  normalize_list(file_content["GWL"])
    file_content["P"] =    normalize_list(file_content["P"])
    file_content["T"] =    normalize_list(file_content["T"])
    file_content["ET"] =   normalize_list(file_content["ET"])
    file_content["NDVI"] = normalize_list(file_content["NDVI"])
    return file_content

def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def save_ccc(path, file_content):
    mat = []
    for var1 in ["GWL", "P", "T", "ET", "NDVI"]:
        mat.append([ccc(file_content[var1], file_content[var2]) for var2 in ["GWL", "P", "T", "ET", "NDVI"]])
     
    df = pd.DataFrame(mat, index=["GWL", "P", "T", "ET", "NDVI"], columns=["GWL", "P", "T", "ET", "NDVI"])
    df.to_csv(path)


for file in get_files_name()[:20]:
    file_content = pd.read_csv(f"data/{file}.csv")    
    make_graph(file_content, f"part1/img/std/{file}.png")
    
    save_ccc(f"part1/ccc/std/{file}.csv", file_content)

    file_content = normalize_file(file_content)
    make_graph(file_content, f"part1/img/norm/{file}.png")
    
    save_ccc(f"part1/ccc/norm/{file}.csv", file_content)