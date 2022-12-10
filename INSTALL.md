## Installation

### Install UA-CMDet

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n UA-CMDet python=3.7 -y
source activate UA-CMDet

conda install cython
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

c. Clone the UA-CMDet repository.

```shell
git clone https://github.com/SunYM2020/UA-CMDet.git
cd UA-CMDet
```

d. Compile cuda extensions.

```shell
bash compile.sh
```

e. Install UA-CMDet (other dependencies will be installed automatically).

```shell
pip install -r requirements.txt
python setup.py develop
# or "pip install -e ."
```

Note:

1. It is recommended that you run the step e each time you pull some updates from github. If there are some updates of the C/CUDA codes, you also need to run step d.
The git commit id will be written to the version number with step e, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, UA-CMDet is installed on `dev` mode, any modifications to the code will take effect without installing it again.


f. Compile polyiou.
```
sudo apt-get install swig
cd eval
swig -c++ -python polyiou.i
```