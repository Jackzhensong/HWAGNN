#!/bin/sh
# unzip data.zip
# pip install -r requirement.yaml
# run Reproduce.sh

## haar
python main.py --dataset photo --lr 0.01 --wd 0.001 --dropout 0.6 --hidden 512 --num_layers 3 --alpha 0.9
python main.py --dataset computers --lr 0.01 --wd 0.001 --dropout 0.5 --hidden 512 --num_layers 2 --alpha 0.7
python main.py --dataset cs --lr 0.01 --wd 0.005 --dropout 0.7 --hidden 512 --num_layers 3 --alpha 0.5
python main.py --dataset physics --lr 0.01 --wd 0.001 --dropout 0.3 --hidden 512 --num_layers 2 --alpha 0.7

python main.py --dataset chameleon_filtered --lr 0.1 --wd 0.0001 --dropout 0.7 --num_layers 3 --alpha 0.3
python main.py --dataset squirrel_filtered --lr 0.1 --wd 0.001 --dropout 0.7 --num_layers 2 --alpha 0.1
python main.py --dataset film --lr 0.1 --wd 0.005 --dropout 0.5 --hidden 512 --num_layers 3 --alpha 0.3
python main.py --dataset deezer --lr 0.01 --wd 0.005 --dropout 0.3 --hidden 512 --num_layers 2 --alpha 0.3