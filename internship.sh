#!/bin/bash

# Script to be run via Slurm for Jérôme CHEN's 2024 Internship

#SBATCH --job-name=Australia_GWL_Model
#SBATCH --ntasks=1
#SBATCH --
#SBATCH --
#SBATCH --

na_threshhold=0.15

seed=20

number_files=1
batch_size=5
number_layer=3
layer_size=6
dropout=0.15
nb_epoch=5000
learning_rate=0.001

echo "____________ Part a ____________"

# python3 make_data.py -na $na_threshhold --mean month
# python3 part_a/main.py --sub_path=part_a

echo "____________ Part b ____________"


# change the batch size
# python3 make_model.py --name L1  -ff 100 -bs 5  -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 4-1 -d $dropout -lr $learning_rate -lf l1  --seed 164583
# python3 make_model.py --name MSE -ff 100 -bs 10 -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 4-1 -d $dropout -lr $learning_rate -lf mse --seed 164583
# python3 make_model.py --name test_3 -ff 100 -bs 20 -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 4-1 -d $dropout -lr $learning_rate
echo "Making data standard"
python3 make_data.py -na $na_threshhold --mean month -q

echo "Making model"
python3 make_model.py --name 4-1_standard -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 4-1 -d $dropout -lr $learning_rate 
python3 make_model.py --name 4-6_standard -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 4-6 -d $dropout -lr $learning_rate
python3 make_model.py --name 5-6_standard -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 5-6 -d $dropout -lr $learning_rate

echo "Making data normalized"
python3 make_data.py -na $na_threshhold -n --mean month -q

echo "Making model"
python3 make_model.py --name 4-1_normalized -ff 100 -bs 50 -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 4-1 -d $dropout -lr $learning_rate
python3 make_model.py --name 4-6_normalized -ff 100 -bs 50 -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 4-6 -d $dropout -lr $learning_rate
python3 make_model.py --name 5-6_normalized -ff 100 -bs 50 -ne $nb_epoch -nl $number_layer -ls $layer_size -mt 5-6 -d $dropout -lr $learning_rate

# echo "____________ Part c ____________"
# echo "____________ Part d ____________"
# echo "____________ Part e ____________"

# python3 make_data.py -na $na_threshhold -q -m month -n
# python3 make_model.py -ff $number_files --seed $seed -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf l1 --name month_normalized_l1
# python3 make_model.py -ff $number_files --seed $seed -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf mse --name month_normalized_mse

# python3 make_data.py -na $na_threshhold -q -m month
# python3 make_model.py -ff $number_files --seed $seed -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf l1 --name month_raw_l1
# python3 make_model.py -ff $number_files --seed $seed -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf mse --name month_raw_mse

# python3 make_data.py -na $na_threshhold -q -m file -n
# python3 make_model.py -ff $number_files --seed $seed -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf l1 --name file_normalized_l1
# python3 make_model.py -ff $number_files --seed $seed -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf mse --name file_normalized_mse

# python3 make_data.py -na $na_threshhold -q -m file
# python3 make_model.py -ff $number_files --seed $seed -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf l1 --name file_raw_l1
# python3 make_model.py -ff $number_files --seed $seed -bs $batch_size -s -d $dropout -ne $nb_epoch -lr $learning_rate -nl $number_layer -ls $layer_size -lf mse --name file_raw_mse

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
