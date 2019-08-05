import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def conv_relu(channels_in, channels_out, filter_size, stride):
    return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, filter_size, stride),
            nn.LeakyReLU(inplace=True))


def conv_base(input_shape):
    return nn.Sequential(
            conv_relu(input_shape[2], 8, 8, 4),
            conv_relu(8, 16, 5, 2),
            conv_relu(16, 32, 5, 2))


def fc_base(input_dim, hidden_dim, last=True):
    layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            ]
    if last:
        layers.append(nn.Linear(hidden_dim, 1))

    return nn.Sequential(*layers)


def get_shape(layer, input_shape):
    """
    Assunes input_shape is [h, w, c].
    """
    result = 1

    for mult in  layer(torch.zeros([1] + input_shape[::-1])).shape:
        result *= mult

    return result


def ValueNetwork(input_shape, hidden_dim):
    conv = conv_base(input_shape)
    fc = fc_base(get_shape(conv, input_shape), hidden_dim)

    return nn.Sequential(conv, fc)


class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim):
        super().__init__()

        self.conv1 = conv_base(input_shape)
        self.fc1 = fc_base(
                get_shape(self.conv1, input_shape) + num_actions,
                hidden_dim, last=False)

        self.conv2 = conv_base(input_shape)
        self.fc2 = fc_base(
                get_shape(self.conv2, input_shape) + num_actions,
                hidden_dim, last=False)

        self.apply(weights_init_)

    def forward(self, state, action):
        x1 = self.conv1(state).view(state.shape[0], -1)
        x1 = torch.cat([x1, action], 1)
        x1 = self.fc1(x1)

        x2 = self.conv2(state).view(state.shape[0], -1)
        x2 = torch.cat([x2, action], 1)
        x2 = self.fc2(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim):
        super().__init__()

        self.conv = conv_base(input_shape)
        self.fc = fc_base(get_shape(self.conv, input_shape), hidden_dim, last=False)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.conv(state)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


# Not implemented.
class DeterministicPolicy(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(input_shape, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean


    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean
    
