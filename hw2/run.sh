#!/bin/bash

# 1
python run.py corner-keypoint --sigma 6.0 --threshold 16
python run.py ransac-linefit --num-lines 4 -ns 100 -t 2

# 2
# python run.py corner-keypoint --sigma 5.0 --threshold 25
# python run.py ransac-linefit --num-lines 4 -ns 100 -t 2

# 3
# python run.py corner-keypoint --sigma 8.0 --threshold 15
# python run.py ransac-linefit --num-lines 4 -ns 100 -t 2