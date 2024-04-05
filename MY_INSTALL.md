## Installation

### Install UA-CMDet

a. Create a conda virtual environment and activate it. Then install Cython.
```shell
conda create -n UA-CMDet python=3.7 -y
source activate UA-CMDet

conda install cython
```
b. Install PyTorch and torchvision
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

c. Clone github
```shell
git clone https://github.com/SunYM2020/UA-CMDet.git
cd UA-CMDet
```

d. Compile cuda extensions.
```shell
bash compile.sh
pip install -U openmim
mim install mmcv==0.4.3

```

comment mmcv ở cái requirements.txt lại
e. Install UA-CMDet
```shell
pip install -r requirements.txt
python setup.py develop
```
f. Compile polyiou
```shell
sudo apt-get install swig
cd eval
swig -c++ -python polyiou.i

pip install "pillow<7"
cd eval
python setup.py build_ext –inplace
```
Nếu thiếu setup.py, cop ở cái DOTA_devkit sang
Sau đó cop cái file UA-CMDet/DOTA_devkit/_polyiou.cpython-37m-x86_64-linux-gnu.so từ eval sang DOTA_devkit
