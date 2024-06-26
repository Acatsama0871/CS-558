#!/bin/bash

# problem 1
python run.py k-means-segmentation -i data/white-tower.png -o result -n 10

# problem 2
python run.py slic -i data/white-tower.png -o result -f 50 -m 10 -max 10
python run.py slic -i data/wt_slic.png -o result -f 50 -m 10 -max 10

# problem 3
python run.py sky-classification -tm data/sky/train_sky_masked.jpg -o result/sky_output -n 10
