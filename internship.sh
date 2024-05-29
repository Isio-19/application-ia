#!/bin/bash

# Script to be run via Slurm for Jérôme CHEN's 2024 Internship

#SBATCH --job-name=Australia_GWL_Model
#SBATCH --ntasks=1
#SBATCH --
#SBATCH --
#SBATCH --

na_threshhold=0.15

number_files=20
batch_size=5
number_layer=4
layer_size=6
dropout=0.15
nb_epoch=1000
learning_rate=0.001

# python3 make_data.py -na $na_threshhold -q -m month -n
# python3 make_model.py -ff $number_files --seed 20 -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf l1 --name month_normalized_l1
# python3 make_model.py -ff $number_files --seed 20 -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf mse --name month_normalized_mse

# python3 make_data.py -na $na_threshhold -q -m month
# python3 make_model.py -ff $number_files --seed 20 -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf l1 --name month_raw_l1
# python3 make_model.py -ff $number_files --seed 20 -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf mse --name month_raw_mse

# python3 make_data.py -na $na_threshhold -q -m file -n
# python3 make_model.py -ff $number_files --seed 20 -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf l1 --name file_normalized_l1
# python3 make_model.py -ff $number_files --seed 20 -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf mse --name file_normalized_mse

# python3 make_data.py -na $na_threshhold -q -m file
# python3 make_model.py -ff $number_files --seed 20 -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf l1 --name file_raw_l1
# python3 make_model.py -ff $number_files --seed 20 -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf mse --name file_raw_mse

# echo -n "Training with params: "
# echo -n "number_files: $number_files "
# echo -n "batch_size: $batch_size "
# echo -n "dropout: $dropout "
# echo -n "nb_epoch: $nb_epoch "
# echo -n "learning_rate: $learning_rate "
# echo
# for number_layer in $(seq 2 6)
# do
#     for layer_size in $(seq 4 10)
#     do
#         echo "Training with params number_layer: $number_layer, layer_size $layer_size "
#         python3 make_model.py -ff $number_files -bs $batch_size -nl $number_layer -ls $layer_size -ne $nb_epoch -lr $learning_rate
#     done
# done
