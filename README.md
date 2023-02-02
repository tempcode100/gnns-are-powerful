# Representation Power of Graph Neural Networks: Improved Expressivity Characterization via Algebraic Analysis

This directory includes the pytorch implementation for the experiments conducted
in the paper "Representation Power of Graph Neural Networks: Improved Expressivity Characterization via Algebraic Analysis". To test
the provided code please install the latest stable Pytorch version from 
"https://pytorch.org/get-started/locally/". Also install the latest 
Pytorch Geometric CUDA version from
"https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html".For
example try:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
and
```
conda install pyg -c pyg
```
Also install the dependencies via:
```
pip install -r requirements.txt
```
To execute the experiments of section 5 run:

```
python main_Fig_1.py
```
```
python main_Fig_2.py
```
To execute the experiments of section 7 run:
```
python main_CSL.py
```
and
```
python main.py
```
```
python main_GIN_1.py
```
```
python main_GIN_plus.py
```
