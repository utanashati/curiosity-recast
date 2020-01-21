import gym
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces.box import Box
from gym.spaces import Discrete
import os
import datetime

from colors import same_1


class PicolmazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_rooms=4, colors_func=same_1):
        pic_size = 42
        package_dir, _ = os.path.split(__file__)

        pic_names = [
            os.path.join(
                package_dir, f"../pics/{pic_size}/{i}.jpg"
            ) for i in range(num_rooms)
        ]
        pics = [imread(pic_name, as_gray=True) for pic_name in pic_names]

        cmaps_orig = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        ]

        def pick_color(size):
            assert size < len(cmaps_orig) + 1
            return list(np.random.choice(cmaps_orig, size, replace=False))

        def apply_cmaps(cmaps, pic):
            norm = plt.Normalize(vmin=0, vmax=256)
            return [np.moveaxis(
                getattr(plt.cm, cmap)(norm(pic))[:, :, :3], -1, 0
            ).astype(np.float32) for cmap in cmaps]

        rooms = range(num_rooms)
        # colors = [2**i for i in rooms]  # Different entropy levels
        # colors = range(1, num_rooms + 1)
        # colors = [1] * num_rooms
        colors = colors_func(num_rooms)
        print(colors)

        cpics = [
            apply_cmaps(
                pick_color(color), pics[room]
            ) for room, color in zip(rooms, colors)]

        out_dir = os.path.join(package_dir, f"../pics/out")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i, room in enumerate(cpics):
            for j, pic in enumerate(room):
                plt.imsave(os.path.join(
                    package_dir, f"../pics/out/{i}_{j}.jpg"
                ), np.moveaxis(pic, 0, -1))

        # steps_per_episode = 100

        self.num_rooms = num_rooms
        self.colors_func = colors_func
        self.rooms = np.arange(num_rooms).reshape(int(num_rooms**(1 / 2)), -1)
        self.room = np.random.choice(np.arange(num_rooms))
        self.room_2d = list(np.unravel_index(self.room, self.rooms.shape))
        self.cpics = cpics
        self.cpic_inds = [0] * num_rooms
        self.count_end_episode = 0

        self.actions = ['play', 'up', 'right', 'down', 'left']

        self.action_space = Discrete(len(self.actions))
        self.observation_space = Box(0.0, 1.0, (3, pic_size, pic_size))
        # self.action_space = ActionSpace()
        # self.observation_space = StateSpace()

        self.render_flag = True

    def step(self, action):
        assert action in range(len(self.actions))
        action = self.actions[action]

        self.count_end_episode += 1
        self.action = action
        # end_episode = self.count_end_episode > steps_per_episode + 1
        end_episode = False

        if action == 'play':
            inds = np.arange(len(self.cpics[self.room]))
            if inds.size > 1:
                choose_from = np.delete(inds, self.cpic_inds[self.room])
            else:
                choose_from = inds
            self.cpic_inds[self.room] = \
                np.random.choice(choose_from)
            return self.cpics[self.room][self.cpic_inds[self.room]], 0, \
                end_episode, {}

        elif action == 'up':
            if self.room_2d[1] + 1 < self.rooms.shape[1]:
                self.room_2d[1] += 1
                self.room = np.ravel_multi_index(
                    np.array(self.room_2d).T, self.rooms.shape)
            return self.cpics[self.room][self.cpic_inds[self.room]], 0, \
                end_episode, {}

        elif action == 'right':
            if self.room_2d[0] + 1 < self.rooms.shape[0]:
                self.room_2d[0] += 1
                self.room = np.ravel_multi_index(
                    np.array(self.room_2d).T, self.rooms.shape)
            return self.cpics[self.room][self.cpic_inds[self.room]], 0, \
                end_episode, {}

        elif action == 'down':
            if self.room_2d[1] - 1 > -1:
                self.room_2d[1] -= 1
                self.room = np.ravel_multi_index(
                    np.array(self.room_2d).T, self.rooms.shape)
            return self.cpics[self.room][self.cpic_inds[self.room]], 0, \
                end_episode, {}

        elif action == 'left':
            if self.room_2d[0] - 1 > -1:
                self.room_2d[0] -= 1
                self.room = np.ravel_multi_index(
                    np.array(self.room_2d).T, self.rooms.shape)
            return self.cpics[self.room][self.cpic_inds[self.room]], 0, \
                end_episode, {}

    def reset(self):
        self.__init__(
            num_rooms=self.num_rooms, colors_func=self.colors_func)
        return self.step(0)[0]

    def render(self, mode='human', close=False):
        if self.render_flag:
            self.exp_dir = os.path.join(
                os.getcwd(),
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            if not os.path.exists(self.exp_dir):
                os.makedirs(self.exp_dir)
            self.render_flag = False

        print(f"Room: {self.room}\nPic: {self.cpic_inds[self.room]}")

        fig, ax = plt.subplots()
        ax.imshow(
            np.moveaxis(self.cpics[self.room][self.cpic_inds[self.room]], 0, -1),
            interpolation='bicubic')
        ax.axis('off')

        textstr = self.action
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

        plt.savefig(os.path.join(
            self.exp_dir,
            datetime.datetime.now().strftime('%H-%M-%S-%f.png')), dpi=72)

        plt.close()


# class ActionSpace(gym.Space):
#     def sample(self):
#         actions = ['play', 'up', 'right', 'down', 'left']
#         return np.random.choice(actions)

#     def contains(self, x):
#         actions = ['play', 'up', 'right', 'down', 'left']
#         return x in actions

#     def to_jsonable(self, sample_n):
#         """Convert a batch of samples from this space to a JSONable data type."""
#         # By default, assume identity is JSONable
#         return sample_n

#     def from_jsonable(self, sample_n):
#         """Convert a JSONable data type to a batch of samples from this space."""
#         # By default, assume identity is JSONable
#         return sample_n


# class StateSpace(gym.Space):
#     def sample(self):
#         cpics_flat = list(itertools.chain.from_iterable(cpics))
#         return np.random.choice(cpics_flat)

#     def contains(self, x):
#         cpics_flat = list(itertools.chain.from_iterable(cpics))
#         return x in cpics_flat

#     def to_jsonable(self, sample_n):
#         """Convert a batch of samples from this space to a JSONable data type."""
#         # By default, assume identity is JSONable
#         return sample_n

#     def from_jsonable(self, sample_n):
#         """Convert a JSONable data type to a batch of samples from this space."""
#         # By default, assume identity is JSONable
#         return sample_n
