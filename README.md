# curiosity-recast
This code is based on the [A3C implementaiton](https://github.com/ikostrikov/pytorch-a3c) by Ilya Kostrikov.

## Installation
### AWS Deep Learning AMI (Ubuntu 16.04)
```
source activate pytorch_p36
pip install opencv-python gym gym[atari] tensorboard tensorboard_logger
sudo apt-get install libav-tools
git clone https://github.com/utanashati/curiosity-recast.git
cd curiosity-recast
```

On Ubuntu 18.04, do `sudo apt-get install ffmpeg` instead of `sudo apt-get install libav-tools`.