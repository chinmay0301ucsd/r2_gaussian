<!-- <p align="center"> <img src="assets/logo.png" width="250px"> </p> -->
## Vol3DGS rasterizer for Tomopgraphy
This repo is built on top of the [R2-Gaussian repo](https://github.com/Ruyi-Zha/r2_gaussian). Please see the repo link for downloading datasets. For the conda environment setup, use the same environment setup as in the [Vol3DGS repo](https://github.com/chinmay0301ucsd/Vol3DGS).
For this repository also we use the same conda environment setup as in the [Vol3DGS repo](https://github.com/chinmay0301ucsd/Vol3DGS).

## Installation

```
# Download Code
git clone https://github.com/chinmay0301ucsd/r2_gaussian --recursive
cd r2_gaussian
bash install.sh
# Install TIGRE for data generation and initialization
wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip
unzip v2.3.zip
pip install TIGRE-2.3/Python --no-build-isolation
```

## Installing datasets
For installing and setting up the datasets, refer to this [link](https://github.com/Ruyi-Zha/r2_gaussian#2-dataset). 

## Running the code
Install wandb, and login
```
pip install wandb
wandb login
```

Example for training script

```
python3 train_ours.py -s <data path> --render_backend slang --wandb -m <path to trained model>
# --render_backend can be r2, slang for running the baseline and our code respectively
```

For evaluation, 
```
python3 test.py -m <path to trained model>
```
The`scripts_bash` folder has more example scripts 

## 5. Acknowledgements

Our code is adapted from [R2-Gaussian repo](https://github.com/Ruyi-Zha/r2_gaussian). We sincerely thank the authors for their excellent codebase.
This project follows the same license as [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

If you find our work and code useful, please cite this work and ⭐ the repo:

```@misc{talegaonkar2025vol3dgs,
      title={Volumetrically Consistent 3D Gaussian Rasterization},
      author={Chinmay Talegaonkar and Yash Belhe and Ravi Ramamoorthi and Nicholas Antipa},
      year={2025},
      eprint={2412.03378},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
