## SOWdgl: Robust Stochastic Open-world Learning for Dynamic Graphs

This repository contains the author's implementation Pytorch in  paper "Robust Stochastic Open-world Learning for Dynamic Graphs".


## Dependencies

- Python (>=3.6)
- torch:  (>= 1.7.0)
- numpy (>=1.17.4)
- sklearn

## Datasets
We provide link for datasets of DBLP3, DBLP5, Brain and Reddit here:https://drive.google.com/drive/folders/1u0pZjAFA6zRS2ePxkdvXdjCunrA-jg_k

## Implementation

Here we provide the implementation of OpenWGL, along with the default dataset (DBLP5). The repository is organised as follows:

 - `data/` contains the necessary dataset files and config files;
 - `method/` contains the implementation of the SOWdgl and the basic utils;

 Finally, `main.py` puts all of the above together and can be used to execute a full training run on the datasets.

## Process
 - Place the datasets in `data/`
 - Training/Testing:
 ```bash
 python main.py
 ```
 
