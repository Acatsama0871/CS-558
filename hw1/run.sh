#!/bin/bash

# generate the gaussian filtered images
#!/bin/bash

# gaussian filter
python run.py gaussian-filter -i "data/kangaroo.pgm" -sig 1
python run.py gaussian-filter -i "data/kangaroo.pgm" -sig 2
python run.py gaussian-filter -i "data/plane.pgm" -sig 1
python run.py gaussian-filter -i "data/plane.pgm" -sig 3
python run.py gaussian-filter -i "data/red.pgm" -sig 1
python run.py gaussian-filter -i "data/red.pgm" -sig 2

# sobel operator
python run.py sobel-operator-gradient-magnitude -i "data/kangaroo.pgm" -sig 2 -t 22.5
python run.py sobel-operator-gradient-magnitude -i "data/plane.pgm" -sig 3 -t 22
python run.py sobel-operator-gradient-magnitude -i "data/red.pgm" -sig 2 -t 22

# non-maximum suppression edge detection
python run.py non-maximum-suppression-edge-detection -i "data/kangaroo.pgm" -sig 3.0 -t 30.0
python run.py non-maximum-suppression-edge-detection -i "data/plane.pgm" -sig 3.0 -t 35.0
python run.py non-maximum-suppression-edge-detection -i "data/red.pgm" -sig 2.0 -t 22.0


