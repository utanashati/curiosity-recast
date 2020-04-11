# Maximum Likelihood Estimator with Variance for Curiosity-Driven Exploration
This code is based on the [A3C implementaiton](https://github.com/ikostrikov/pytorch-a3c) by Ilya Kostrikov.

**TODO**
- [] Give a summary of the project (maybe from report)
- [] Add report
- [] Add link to defense

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
```
python main.py --game "atari" --env-name "PongDeterministic-v4" --num-processes 16 --save-model-again-eps 5 --save-video-again-eps 1 --max-episodes 20 --random-seed --no-curiosity --short-description "pong-nocuriosity" --num-stack 1
```

### [noreward-rl](https://github.com/pathak22/noreward-rl): VizDoom
#### Dense
```
python main.py --num-processes 16 --game "doom" --env-name "dense" --time-sleep 60 --save-model-again-eps 5 --save-video-again-eps 1 --max-episodes 250 --short-description "doom-curiosity"
```

### Picolmaze
For Picolmaze, we did not train an RL algorithm, just the inverse and then forward models, and compared the baseline to the one with a new loss.

To train the **inverse model** for 9 rooms with a periodic arena in 'ascending entropies' setting:
```
python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-diff-periodic-same-env" --beta 0 --num-rooms 9 --colors "diff_1_num_rooms" --periodic
```

Same for deterministic setting:
```
python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-same-1-periodic-same-env" --beta 0 --num-rooms 9 --colors "same_1" --periodic
```

Same for 8 options per room:
```
python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-same-8-periodic-same-env" --beta 0 --num-rooms 9 --colors "same_8" --periodic
```

Now, to train the **baseline forward model** for the same settings in the same order:
```
python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-diff-periodic-same-env-forw" --beta 1 --num-rooms 9 --colors "diff_1_num_rooms" --curiosity-file "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-diff-periodic-same-env)/models/curiosity_XXXX.XX.XX-XX.XX.XX_XXXXXX.pth" --periodic --env-folder "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-diff-periodic-same-env)"

python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-same-1-periodic-same-env-forw" --beta 1 --num-rooms 9 --colors "same_1" --curiosity-file "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-same-1-periodic-same-env)/models/curiosity_XXXX.XX.XX-XX.XX.XX_XXXXXX.pth" --periodic --env-folder "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-same-1-periodic-same-env)/"

python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-same-8-periodic-same-env-forw" --beta 1 --num-rooms 9 --colors "same_8" --curiosity-file "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-same-8-periodic-same-env)/models/curiosity_XXXX.XX.XX-XX.XX.XX_XXXXXX.pth" --periodic --env-folder "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-same-8-periodic-same-env)/"
```

Use the inverse model file you got as input.

Note that for the inverse model, `beta = 0`, and now `beta = 1`, following the equation
<img src="https://render.githubusercontent.com/render/math?math=%5Cmin%5Climits_%7B%5Ctheta_P%2C%20%5Ctheta_I%2C%20%5Ctheta_F%7D%20%5Cleft%5B%20-%5Clambda%20%5Cmathbb%7BE%7D_%7B%5Cpi%7D%5B%5Csum_t%20r_t%5D%20%2B%20(1%20-%20%5Cbeta)%20L_I%20%2B%20%5Cbeta%20L_F%20%5Cright%5D"> from [Pathak et al.](https://github.com/pathak22/noreward-rl) (`beta == 0` <=> only the inverse model is being trained, `beta == 1` <=> only the forward model is being trained, <img src="https://render.githubusercontent.com/render/math?math=\lambda = 0">).

Finally, to train the **modified forward model** for the same settings in the same order:
```
python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-diff-periodic-same-env-forw" --beta 1 --num-rooms 9 --colors "diff_1_num_rooms" --curiosity-file "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-diff-periodic-same-env)/models/curiosity_XXXX.XX.XX-XX.XX.XX_XXXXXX.pth" --periodic --env-folder "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-diff-periodic-same-env)" --new-curiosity

python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-same-1-periodic-same-env-forw" --beta 1 --num-rooms 9 --colors "same_1" --curiosity-file "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-same-1-periodic-same-env)/models/curiosity_XXXX.XX.XX-XX.XX.XX_XXXXXX.pth" --periodic --env-folder "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-same-1-periodic-same-env)/" --new-curiosity

python main_uniform.py --num-processes 1 --time-sleep 20 --save-model-again-eps 5 --max-episodes 100 --short-description "uniform-9-same-8-periodic-same-env-forw" --beta 1 --num-rooms 9 --colors "same_8" --curiosity-file "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-same-8-periodic-same-env)/models/curiosity_XXXX.XX.XX-XX.XX.XX_XXXXXX.pth" --periodic --env-folder "runs/picolmaze/XXXX.XX.XX-XX.XX.XX(uniform-9-same-8-periodic-same-env)/" --new-curiosity
```
