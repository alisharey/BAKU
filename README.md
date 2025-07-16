# BAKU: An Efficient Transformer for Multi-Task Policy Learning

This is a repository containing the code for the paper [BAKU: An Efficient Transformer for Multi-Task Policy Learning](https://arxiv.org/abs/2406.07539).

![intro](https://github.com/siddhanthaldar/baku-release/assets/25313941/7df30d79-6864-4b39-bd33-55376829b28e)

## Installation Instructions

In order to install the required dependencies, please follow the instructions provide [here](Instructions.md).

## Access to Datasets
We have added the instructions for running BAKU on the LIBERO benchmark [here](Instructions.md). For access to the datasets for Meta-World, DMControl, and the real world xArm Kitchen, please send an email to the sh6474@nyu.edu. 

## Bibtex
If you find this work useful, please cite the paper using the following bibtex:
```
@article{haldar2024baku,
  title={BAKU: An Efficient Transformer for Multi-Task Policy Learning},
  author={Haldar, Siddhant and Peng, Zhuoran and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2406.07539},
  year={2024}
}
```
# Install libgpg-error (newer version for compatibility)

conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3


conda install -c menpo osmesa=12.2.2.dev

wget https://www.gnupg.org/ftp/gcrypt/libgpg-error/libgpg-error-1.47.tar.bz2
tar xjf libgpg-error-1.47.tar.bz2
cd libgpg-error-1.47
./configure --prefix=$CONDA_PREFIX 
make
make install

# Install the missing gpg-error-config script
cd src
make gpg-error-config
cp gpg-error-config $CONDA_PREFIX/bin/
chmod +x $CONDA_PREFIX/bin/gpg-error-config
cd ../..

# Set up environment variables
export PATH=$CONDA_PREFIX/bin:$PATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify gpg-error installation
which gpg-error-config
gpg-error-config --version

wget https://www.gnupg.org/ftp/gcrypt/libgcrypt/libgcrypt-1.5.3.tar.gz
tar xzf libgcrypt-1.5.3.tar.gz
cd libgcrypt-1.5.3
./configure --prefix=$CONDA_PREFIX
make
make install

# Set up library paths
<!-- export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include --> probably not needed

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# pip install pyrender -> might not be needed
 

# suggestions from here
# https://pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html
 
 

