#!/bin/bash

# # key point detection
python run.py corner-keypoint --sigma 6.0 --threshold 16
# # ransac line fitting
python run.py ransac-linefit --num-lines 4 -ns 100 -t 2
# hough line fitting
python run.py hough-transform --num-lines 4 --theta-step 2.0 --rho-step 2.0 --min-dist 10
# hough line fitting half step
python run.py hough-transform --num-lines 4 --theta-step 1.0 --rho-step 1.0 --min-dist 10
# hough line fitting double step
python run.py hough-transform --num-lines 4 --theta-step 4.0 --rho-step 4.0 --min-dist 10
