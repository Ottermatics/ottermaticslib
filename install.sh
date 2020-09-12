sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update

sudo apt-get install git
sudo apt install gcc
sudo apt install g++
sudo apt-get install build-essential


sudo apt-get install ffmpeg

# Install necessary system packages
sudo apt-get install -y \
    ffmpeg \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev

# Install gstreamer for audio, video (optional)
sudo apt-get install -y \
    libgstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good

echo 'Installing Python3'
sudo apt-get install python3 python3-pip python3-dev python3-setuptools
python3 -m pip install --upgrade pip
pip3 install virtualenv



echo 'Installing Anaconda Python (follow instructions, agree & yes)'
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh

conda create -n py3
conda activate py3
conda install -c anaconda pip
pip install -U pip-tools

echo 'Setting Up Github Account (first input)'
git config --global user.email $1
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa

echo 'Add your public key to your github account'
cat < ./.ssh/id_rsa.pub

echo 'Installing Ottermatics Lib'
git clone git@github.com:SoundsSerious/ottermaticslib.git
cd ottermaticslib
python -m pip install -r requirements.txt
python setup.py install

 