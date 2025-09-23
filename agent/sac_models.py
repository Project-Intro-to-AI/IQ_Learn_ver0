import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions as pyd
from torch.autograd import Variable, grad

import utils.utils as utils

# Initialize Policy weights
def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)

        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action, both=True)
        ones = torch.ones(q[0].size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=[ones, ones],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class DoubleQCriticMax(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCriticMax, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)

        if both:
            return q1, q2
        else:
            return torch.max(q1, q2)


class ConvQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(ConvQCritic, self).__init__()
        # Assuming obs_dim is the dimension of flattened image vector,
        # it needs to be reshaped back to (C, H, W) format.
        # Example: (96*96*3) should be reshaped to (3, 96, 96).
        channels, height, width = 3, 96, 96  # Adjust if necessary

        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for the fully connected layers
        )

        # Compute the size of the flattened features after convolutional layers
        with torch.no_grad():
            self.feature_dim = self.conv_layers(torch.zeros(1, channels, height, width)).shape[1]

        # Q architecture
        self.Q = utils.mlp(self.feature_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        # Assuming obs is a batch of image tensors with shape (B, C, H, W)
        conv_out = self.conv_layers(obs)
        obs_action = torch.cat([conv_out, action], dim=-1)
        q = self.Q(obs_action)

        return q


class SingleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, from_pixels=False, args=None):
        super(SingleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.from_pixels = from_pixels
        self.args = args

        if from_pixels:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten()  # Flatten the convolutional layer output
            )
            # Compute the output size of the CNN (for 84x84 input)
            with torch.no_grad():
                self.cnn_output_size = self.cnn(torch.zeros(1, 3, 84, 84)).shape[1]

            # Trunk for CNN output + action
            self.trunk = utils.mlp(self.cnn_output_size + action_dim, hidden_dim, 1, hidden_depth)
            self.cnn = self.cnn
        else:
            # Trunk for state + action
            self.trunk = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
            self.cnn = None

        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        if self.from_pixels and self.cnn is not None:
            # Resize to 84x84 if needed, and transpose to channels-first (B, C, H, W)
            if obs.shape[1] != 3:  # If not channels-first (check if H != 3)
                obs = F.interpolate(obs.permute(0, 3, 1, 2), size=(84, 84), mode='bilinear')  # (B, C, H, W) 84x84
            cnn_out = self.cnn(obs)  # CNN out (B, cnn_output_size)
            obs = cnn_out
        # else: obs remains as is (flattened state)

        obs_action = torch.cat([obs, action], dim=-1)
        q = self.trunk(obs_action)

        if self.args.method.tanh:
            q = torch.tanh(q) * 1/(1-self.args.gamma)

        return q, q  # Giả lập double Q bằng cách return hai lần q

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        # Note: This may need adjustment for from_pixels, but assuming states are preprocessed similarly
        if self.from_pixels:
            # For simplicity, assume obs1/obs2 are already CNN-processed or handle inside forward
            # But to keep it simple, we'll process inside forward if needed, but grad_pen uses raw?
            # This might require more work; for now, assume flattened or adjust accordingly
            pass  # Implement if needed, or skip for pixels case

        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(q.size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class DoubleQCriticState(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = utils.mlp(obs_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = utils.mlp(obs_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)

        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = obs1
        policy_data = obs2

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(q[0].size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=[ones, ones],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds, from_pixels=False):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.action_dim = action_dim
        self.from_pixels = from_pixels

        if from_pixels:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten()  # Flatten the convolutional layer output
            )
            # Compute the output size of the CNN (for 84x84 input)
            with torch.no_grad():
                self.cnn_output_size = self.cnn(torch.zeros(1, 3, 84, 84)).shape[1]

            # Trunk for CNN output
            self.trunk = utils.mlp(self.cnn_output_size, hidden_dim, 2 * action_dim, hidden_depth)
        else:
            # Trunk for state input
            self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
            self.cnn = None

        self.outputs = dict()
        self.apply(orthogonal_init_)

    def forward(self, obs):
        if self.from_pixels and self.cnn is not None:
            # Resize to 84x84 if needed, and transpose to channels-first (B, C, H, W)
            if obs.shape[1] != 3:  # If not channels-first (check if H != 3)
                obs = F.interpolate(obs.permute(0, 3, 1, 2), size=(84, 84), mode='bilinear')  # (B, C, H, W) 84x84
            cnn_out = self.cnn(obs)  # CNN out (B, cnn_output_size)
            mu, log_std = self.trunk(cnn_out).chunk(2, dim=-1)
        else:
            cnn_out = obs  # State input
            mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # self.outputs['mu'] = mu
        # self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob, dist.mean