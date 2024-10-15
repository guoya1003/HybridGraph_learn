# HybridGraph: Multi-level Urban Street Representation for Traffic Classification

This repository contains the implementation of a dynamic graph representation framework based on dual spatial semantics for urban street representation and traffic classification (one downstream task example). Our method leverages street-view imagery and social media data to enhance urban traffic analysis.

## Overview

Our approach, Hybrid Graph Neural Network, captures both close-range and long-range spatial relationships, leading to significant improvements in traffic prediction. The framework addresses two key challenges:

1. Considering spatial adjacency between visual elements within street scenes.
2. Accounting for spatial dependency and interaction between streets.

![image](https://github.com/user-attachments/assets/dc050c4c-690c-4615-bd0d-508041cb0137)

## Features

- Dynamic parsing of visual elements within street scenes
- Construction of spatial weight matrices integrating spatial dependency and interaction
- Spatial interpretability analysis tool for downstream tasks
- Improved vehicle speed and flow estimation

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- torch_geometric
- Other dependencies (list them here)

### Installation

1. Clone the repository:
2. git clone https://github.com/yemanzhongting/HybridGraph.git
3. Install the required packages:
pip install -r requirements.txt


### Usage

1. Prepare your Street View Imagery (SVI) data.
   
![1728991299254](https://github.com/user-attachments/assets/bf1cc478-e3a9-4515-a1ef-85ade57ee856)

2. Run the preprocessing script to generate graph representations:
python StreetView_Graph.py --input_dir /path/to/svi --output_dir /path/to/output
3. Train and  Perform the model: python HybridGraph.py


## Results

Our method achieves accuracies of 57.72% and 60.13% in speed and flow predictions, outperforming traditional models by 2.4% and 6.4% respectively.

## Citation

If you use this code in your research, please cite our paper:

Zhang, Y., Li, Y., & Zhang, F. (2024). Multi-level urban street representation with street-view imagery and hybrid semantic graph. ISPRS Journal of Photogrammetry and Remote Sensing. Elsevier.


## License

This project is licensed under the MIT License.


## Contact

For any questions or concerns, please open an issue or contact the repository owner (sggzhang@whu.edu.cn).
This README provides an overview of the project, instructions for installation and usage, information about the results, citation details, and other necessary information. You may want to adjust the content based on the specific structure and requirements of your project.
