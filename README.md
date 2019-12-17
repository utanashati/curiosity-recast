# curiosity-recast
This code is based on the [A3C implementaiton](https://github.com/ikostrikov/pytorch-a3c) by Ilya Kostrikov.

## Install
### AWS Deep Learning AMI (Ubuntu 16.04)
I used `c5.xlarge` configuration with 16 vCPUs and run A3C with 16 processes.

#### Connect
```
ssh -L localhost:8888:localhost:8888 -i key_pair.pem ubuntu@ec2-2-222-222-222.compute-2.amazonaws.com
```

#### Prepare
```
source activate pytorch_p36
git clone https://github.com/utanashati/curiosity-recast.git

pip install --upgrade pip
pip install opencv-python tensorboard tensorboard_logger

sudo apt-get update
sudo apt-get upgrade
```

*(If you get "Recourse temporarily unavailable", wait until the machine has 2/2 checks (or switch to the other pip installation in the meantime).)*

#### Gym
```
pip install gym gym[atari]
sudo apt-get install libav-tools
```

#### VizDoom
```
sudo apt-get install default-jdk pulseaudio
```

**ZDoom dependencies**
```
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip
```

**Cmake Issue**
```
sudo rm -r /usr/local/bin/cmake
sudo /home/ubuntu/anaconda3/envs/pytorch_p36/bin/pip install vizdoom
```

## Reproduce
### [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c): Pong Deterministic
TODO

### [noreward-rl](https://github.com/pathak22/noreward-rl): VizDoom
TODO