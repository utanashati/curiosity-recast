import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grad_sum(model):
    grad = 0
    for param in model.parameters():
        if param.grad is not None:
            grad += param.grad.sum()
    return grad


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


# Curiosity-driven Exploration by Self-supervised Prediction
# arXiv: https://arxiv.org/abs/1705.05363
class IntrinsicCuriosityModule(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(IntrinsicCuriosityModule, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1)
        )

        size = 256
        self.num_actions = action_space.n
        self.inverse = nn.Sequential(
            nn.Linear(32 * 3 * 3 * 2, size),
            nn.ReLU(),
            nn.Linear(size, self.num_actions)
        )

        self.forw = nn.Sequential(
            nn.Linear(32 * 3 * 3 + self.num_actions, size),
            nn.ReLU(),
            nn.Linear(size, 32 * 3 * 3)
        )

        self.apply(weights_init)
        for i in [0, 2]:
            self.inverse[i].weight.data = normalized_columns_initializer(
                self.inverse[i].weight.data, 0.01)
            self.inverse[i].bias.data.fill_(0)

        for i in [0, 2]:
            self.forw[i].weight.data = normalized_columns_initializer(
                self.forw[i].weight.data, 0.01)
            self.forw[i].bias.data.fill_(0)

        self.train()

    def forward(self, state_old, action, state):
        phi1 = self.head(state_old)
        phi2 = self.head(state)

        phi1 = phi1.view(-1, 32 * 3 * 3)
        phi2 = phi2.view(-1, 32 * 3 * 3)

        g = torch.cat([phi1, phi2], 1)
        inv_out = self.inverse(g)

        action_onehot = torch.zeros(1, self.num_actions)
        action_onehot.scatter_(1, action, 1)

        f = torch.cat([phi1, action_onehot], 1)
        forw_out = self.forw(f)

        return inv_out, forw_out, F.mse_loss(forw_out, phi2.detach())
