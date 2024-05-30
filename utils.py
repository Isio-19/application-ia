import pandas as pd
import numpy as np

def get_files_name():
    file_names = pd.read_csv("OUVRAGES.csv")["Ouvrage"].astype("str")

    return_array = []
    for file in file_names:
        if file[0] == "#":
            continue
        return_array.append(file)

    return return_array

def script_error_print(script):
    match script:
        case "make_data.py":
            print(
                "Wrong usage of the script, should be:\n"
                +"python3 make_model.py [OPTIONS]\n"
                +"\n"
                +"With OPTIONS being:\n"
                +"-na, --na_threshhold FLOAT:\tSets the NA threshhold,\n"
                +"\t\t\t\tshould be followed by a float,\n"
                +"\t\t\t\tused when making the data files\n"
                +"\n"
                +"-n, --normalize:\tNormalize the chosen files,\n"
                +"\t\t\tused when making the data files\n"
                +"\n"
                +"-m, --mean STRING:\tDefine which method to use to replace the NANs,\n"
                +"\t\t\tshould be followed by a float which should be 'mean' or 'month',\n"
                +"\t\t\tused when making the data files\n"
                +"\n"
                +"-q, --quiet:\tUsed to suppress the prints\n"
            )
        case "make_model.py":
            print(
                "Wrong usage of the script, should be:\n"
                +"python3 make_model.py [OPTIONS]\n"
                +"\n"
                +"With OPTIONS being:\n"
                +"-ff, --first_files INTEGER:\tSets the number of files that will be taken into account when training the model,\n"
                +"\t\t\t\tshould be followed by an integer\n"
                +"\n"
                +"-s, --shuffle:\tShuffles the data\n"
                +"\n"
                +"--seed INTEGER:\tSets a seed to the script,\n"
                +"\t\tmakes the training and testing process deterministic,\n"
                +"\t\tshould be followed by an integer\n"
                +"\n"
                +"-bs, --batch_size INTEGER:\tSets the size of the batch when training and testing the model,\n"
                +"\t\t\t\tshould be followed by an integer\n"
                +"\n"
                +"-nl, --number_layer INTEGER:\tSets the number of layers in the model,\n"
                +"\t\t\t\tshould be followed by an integer\n"
                +"\n"
                +"-ls, --layer_size INTEGER:\tSets the size of the hidden layers of the LSTM in the model,\n"
                +"\t\t\t\tshould be followed by an integer\n"
                +"\n"
                +"-d, --dropout FLOAT:\tSets the dropout parameter of the LSTM in the model,\n"
                +"\t\t\t\tshould be followed by a float\n"
                +"\n"
                +"-ne, --nb_epochs INTEGER:\tSets the number of epochs used for the training of the model,\n"
                +"\t\t\t\tshould be followed by an integer\n"
                +"\n"
                +"-lr, --learning_rate FLOAT:\tSets the nubmer of epochs used for the training of the model,\n"
                +"\t\t\t\tshould be followed by an integer\n"
                +"\n"
                +"-q, --quiet:\tUsed to suppress the prints\n"
            )