# Active Learning Framework for 3D Electrode Segmentation - README

## Overview

This repository provides the codebase for the research paper **"Active Learning Framework for Efficient Semi-Supervised Segmentation of Battery Electrode Images"**. The framework is designed to facilitate the segmentation of 3D battery electrodes into distinct phases using X-ray nano-holo-tomography data. The segmentation process is optimized through the use of deep learning and active learning methods, minimizing the amount of manual annotation required while achieving results comparable to fully annotated datasets.

## Key Features

- **Active Learning Framework**: Implements an active learning strategy that selects informative training samples at the pixel level, reducing the amount of annotated data required.
- **Application to Battery Materials**: Used for the segmentation of battery electrode materials such as lithium nickel oxide (LNO) and graphite.

## Repository Structure

- **`src/`**: Contains the source code, including model definitions, training scripts, active learning routines, and utility functions.
- **`notebooks/`**: Jupyter notebooks for visualization, experimentation, and interactive analysis.
- **`sample_config.ini`**: A sample configuration file. This should be copied and renamed to `config.ini` for setting up your custom parameters.
- **`requirements.txt`**: The requirements file

## Getting Started

### Prerequisites

- Python 3.11.3
- Required Python packages (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
## Acknowledgements

This version of the software was developed by Laura Hannemose Rieger (lauri at dtu.dk),  DTU Energy.

Copyright © 2024 Technical University of Denmark
