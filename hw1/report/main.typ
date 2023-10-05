#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "CS588-HW1",
  authors: (
    (name: "Haohang Li", email: "hli113@stevens.edu"),
    (name: "Anish Khilani", email: "akhilani@stevens.edu")
  ),
  date: "October 2, 2023",
)

= README
== Note
- Please see the dependencies in the `pyproject.toml` file. The extract version of the dependencies can be installed with #link("https://python-poetry.org/docs/")[#underline(text(blue)[Poetry])] with `poetry.lock`.

- To the result can be replicated by running the `run.sh` script.

- The program is written as a command line tool. The usage can be found by running `python main.py --help`.
  - The entry point is `python run.py`.
  - For the gaussian filtere funcionality: 
    - Entry point: #underline(text(blue)[python run.py gaussian-filter])
  #align(center + top)[
    #figure(
  image("images/gaussian.png", alt: "Gaussian Filter"),
  caption: [
    Gaussian filter
  ],
) <gaussian_filter>
]
  - For the sobel operator funcionality: 
    - Entry point: #underline(text(blue)[python run.py sobel-operator-gradient-magnitude])
    #align(center + top)[
    #figure(
  image("images/sobel.png", alt: "Sobel Operator"),
  caption: [
    Sobel Operator
  ],
) <sobel_operator>
]

  - For the non-maximum supression edge detection(combined the gaussian filter and sobel operator): 
    - Entry point: #underline(text(blue)[python run.py non-maximum-suppression-edge-detection])
    #align(center + top)[
    #figure(
  image("images/nms.png", alt: "Non Maximum Supression"),
  caption: [
    Non Maximum Supression
  ],
) <nms>
]
    
  
