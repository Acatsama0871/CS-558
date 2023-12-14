#!/bin/bash

# problem 1
# question 1
#python run.py harris-corners -i data -o results/harris_raw -n 1000 -g 5 --no-nms

# # question 2
# python run.py harris-corners -i data -o results/harris_nms -n 1000 -g 5 -s 3 --sigma 5

# # question 3
# python run.py feature-matching -d 160 -s nss -r 20
# python run.py feature-matching -d 160 -s ssd -r 20

# Problem 2
# question 1
#python run.py feature-matching -d 160 -s nss -r 20 --save-pickle --return-all
python run.py generate_correspondence
