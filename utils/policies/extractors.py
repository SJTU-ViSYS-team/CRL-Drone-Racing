from distutils.dist import Distribution

import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy
from typing import Tuple, Callable, Any
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from typing import List, Optional, Type, Union, Dict

from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
from torchvision import models

class CustomBaseFeaturesExtractor(BaseFeaturesExtractor):
    is_recurrent=False


def _get_conv_output(net, shape):
    image = th.rand(1, *shape)
    output = net(image)
    return output.numel()


def create_cnn(
        input_channels: int,
        kernel_size: List[int],
        channel: List[int],
        stride: List[int],
        padding: List[int],
        output_channel: int = 0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        batch_norm: bool = False,
        with_bias: bool = True,
) -> nn.Module:
    # assert len(kernel_size) == len(stride) == len(padding) == len(channel), \
    #     "The length of kernel_size, stride, padding and net_arch should be the same."

    if len(channel) > 0:
        modules = [nn.Conv2d(input_channels, channel[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])]
        if batch_norm:
            modules.append(nn.BatchNorm2d(channel[0]))
        modules.append(activation_fn())
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
    else:
        modules = []

    for idx in range(len(channel) - 1):
        modules.append(nn.Conv2d(channel[idx], channel[idx + 1], kernel_size=kernel_size[idx + 1], stride=stride[idx + 1], padding=padding[idx + 1]))
        if batch_norm:
            modules.append(nn.BatchNorm2d(channel[idx + 1]))
        modules.append(activation_fn())
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

    if output_channel > 0:
        last_layer_channel = channel[-1] if len(channel) > 0 else input_channels
        modules.append(nn.Conv2d(last_layer_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        if batch_norm:
            modules.append(nn.BatchNorm2d(output_channel))
        modules.append(activation_fn())
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

    modules.append(nn.Flatten())
    if squash_output:
        modules.append(nn.Tanh())

    net = nn.Sequential(*modules)
    return net


def create_mlp(
        input_dim: int,
        layer: List[int],
        output_dim: int = 0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        batch_norm: bool = False,
        squash_output: bool = False,
        with_bias: bool = True,

) -> nn.Module:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param layer: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param batch_norm: Whether to use batch normalization or not
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(layer) > 0:
        modules = [nn.Linear(input_dim, layer[0], bias=with_bias)]
        if batch_norm:
            modules.append(nn.BatchNorm1d(layer[0]))
        modules.append(activation_fn())
    else:
        modules = []

    for idx in range(len(layer) - 1):
        modules.append(nn.Linear(layer[idx], layer[idx + 1], bias=with_bias))
        if batch_norm:
            modules.append(nn.BatchNorm1d(layer[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = layer[-1] if len(layer) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
        if batch_norm:
            modules.append(nn.BatchNorm1d(output_dim))
        modules.append(activation_fn())

    if squash_output:
        modules.append(nn.Tanh())
    if len(modules) == 0:
        modules.append(nn.Flatten())

    net = nn.Sequential(*modules)

    return net


def set_recurrent_feature_extractor(cls, input_size, rnn_setting):
    recurrent_alias = {
        "GRU":th.nn.GRU,
    }
    rnn_class = rnn_setting.get("class")
    kwargs = rnn_setting.get("kwargs")
    if isinstance(rnn_class, str):
        rnn_class = recurrent_alias[rnn_class]
    cls.__setattr__("recurrent_extractor", rnn_class(input_size=input_size, **kwargs))
    return kwargs.get("hidden_size")


def set_mlp_feature_extractor(cls, name, observation_space, net_arch, activation_fn):
    layer = net_arch.get("mlp_layer", [])
    features_dim = layer[-1] if len(layer) != 0 else observation_space.shape[0]
    input_dim = observation_space.shape[0] if len(observation_space.shape) == 1 else observation_space.shape[1]

    setattr(cls, name + "_extractor",
            create_mlp(
                input_dim=input_dim,
                layer=net_arch.get("mlp_layer", []),
                activation_fn=activation_fn,
                batch_norm=net_arch.get("bn", False),
            )
            )
    return features_dim


def set_cnn_feature_extractor(cls, name, observation_space, net_arch, activation_fn):
    image_channels = observation_space.shape[0]
    backbone = net_arch.get("backbone", None)
    if backbone is not None:
        image_extractor = cls.backbone_alias[backbone](pretrained=True)
        # replace the first layer to match the input channels
        image_extractor.conv1 = nn.Conv2d(image_channels, image_extractor.conv1.out_channels,
                                          kernel_size=image_extractor.conv1.kernel_size,
                                          stride=image_extractor.conv1.stride,
                                          padding=image_extractor.conv1.padding,
                                          bias=image_extractor.conv1.bias is not None)
        if net_arch.get("mlp_layer", None) is not None and len(net_arch.get("mlp_layer", [])) > 0:
            image_extractor.fc = create_mlp(
                input_dim=image_extractor.fc.in_features,
                layer=net_arch.get("mlp_layer"),
                activation_fn=activation_fn,
                batch_norm=False,
            )

    else:
        image_extractor = (
            create_cnn(
                input_channels=image_channels,
                kernel_size=net_arch.get("kernel_size", [5, 3, 3]),
                channel=net_arch.get("channels", [6, 12, 18]),
                activation_fn=activation_fn,
                padding=net_arch.get("padding", [0, 0, 0]),
                stride=net_arch.get("stride", [1, 1, 1]),
                batch_norm=net_arch.get("bn", False)
            )
        )
        _image_features_dims = _get_conv_output(image_extractor, observation_space.shape)
        if net_arch.get("mlp_layer", None) is not None and len(net_arch.get("mlp_layer", [])) > 0:
            image_extractor.add_module("mlp",
                                       create_mlp(
                                           input_dim=_image_features_dims,
                                           layer=net_arch.get("mlp_layer"),
                                           activation_fn=activation_fn,
                                           batch_norm=False,
                                       )
                                       )
    setattr(cls, name + "_extractor", image_extractor)
    cls._image_extractor_names.append(name + "_extractor")
    return _get_conv_output(image_extractor, observation_space.shape)


class TargetExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            net_arch: Dict = {},
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        assert "target" in observation_space.keys()
        layer = net_arch.get("mlp_layer", [])
        self._features_dim = layer[-1] if len(layer) != 0 else observation_space["target"].shape[0]
        super(TargetExtractor, self).__init__(observation_space=observation_space,
                                              features_dim=self.features_dim)
        set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch, activation_fn)

    def forward(self, observations):
        return self.target_extractor(observations['target'])


class StateExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            net_arch: Optional[Dict] = {},
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        assert "state" in list(observation_space.keys())
        self._features_dim = 1
        super(StateExtractor, self).__init__(observation_space=observation_space,
                                             features_dim=self.features_dim)
        feature_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch, activation_fn)
        self._features_dim = feature_dim

    def forward(self, observations):
        return self.state_extractor(observations['state'])


class ImageExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        for key in observation_space.keys():
            assert key in "image" in key or "color" in key or "depth" in key

        # Default feature dimension is 1
        self._features_dim = 1
        super(ImageExtractor, self).__init__(observation_space=observation_space,
                                             features_dim=self.features_dim)
        # Convolution layers for image observations
        self._image_features_dims = []
        self.image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn)

        self._features_dim = sum(self._image_features_dims)

    def forward(self, observations):
        features = []
        for name in self.image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        combined_features = th.cat(features, dim=1)

        return combined_features


class StateTargetExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        # for key in observation_space.keys():
        #     assert key in "state" in key or "target" in key
        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) and ("target" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateTargetExtractor, self).__init__(observation_space=observation_space,
                                                   features_dim=self.features_dim)

        state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)

        self._features_dim = state_features_dim + target_features_dim

    def forward(self, observations):
        return th.cat([self.state_extractor(observations["state"]), self.target_extractor(observations["target"])], dim=1)


class StateTargetImageExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
               and ("target" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateTargetImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)

        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim + _target_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        target_features = self.target_extractor(observations['target'])
        features = [state_features, target_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge state/target features and image features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class ActionImageExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("pastAction" in obs_keys) \
               and ("index" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        # The name must match the extractor defined below, e.g. pastAction_extractor
        _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)

        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _action_features_dim + _index_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [action_features, index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)
        
class ActionImageMaskExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
                and ("index" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys or "mask" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionImageMaskExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        # _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        # Convolution layers for image/mask observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key or "mask" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim =  _index_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        # state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)
        
class ActionImageMaskNoiseExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("pastAction" in obs_keys) \
            and ("noise_target" in obs_keys) \
                and("index" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys or "mask" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionImageMaskNoiseExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        # The name must match the extractor defined below, e.g. pastAction_extractor
        _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _noise_target_features_dim = set_mlp_feature_extractor(self, "noise_target", observation_space["noise_target"], net_arch.get("noise_target", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        # Convolution layers for image/mask observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key or "mask" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _action_features_dim + _index_features_dim + _noise_target_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        action_features = self.pastAction_extractor(observations['pastAction'])
        noise_target_features = self.noise_target_extractor(observations['noise_target'])
        index_features = self.index_extractor(observations['index'])
        features = [action_features, noise_target_features, index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)


class ActionImageMaskNoiseIndexExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("pastAction" in obs_keys) \
            and ("noise_target" in obs_keys) \
                and ("index" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys or "mask" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionImageMaskNoiseIndexExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        # The name must match the extractor defined below, e.g. pastAction_extractor
        _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _noise_target_features_dim = set_mlp_feature_extractor(self, "noise_target", observation_space["noise_target"], net_arch.get("noise_target", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        
        # Convolution layers for image/mask observations
        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key or "mask" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _action_features_dim + _noise_target_features_dim + _index_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        action_features = self.pastAction_extractor(observations['pastAction'])
        noise_target_features = self.noise_target_extractor(observations['noise_target'])
        index_features = self.index_extractor(observations['index'])
        features = [action_features, noise_target_features, index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class ActionStateImageMaskExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
                and ("index" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys or "mask" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionStateImageMaskExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        # Convolution layers for image/mask observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key or "mask" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim  + _index_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [state_features, index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class ActionStateImageIndexExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
                and ("index" in obs_keys)\
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionStateImageIndexExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim   + _index_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [state_features,  index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class ActionStateTargetImageMaskIndexExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
               and ("target" in obs_keys) \
               and ("index" in obs_keys)\
               and ("pastAction" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys or "mask" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionStateTargetImageMaskIndexExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key or "mask" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim + _target_features_dim + _index_features_dim + _action_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        target_features = self.target_extractor(observations['target'])
        action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [state_features, target_features, action_features, index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class ActionStateTargetImageExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
               and ("target" in obs_keys) \
               and ("pastAction" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionStateTargetImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim + _target_features_dim + _action_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        target_features = self.target_extractor(observations['target'])
        action_features = self.pastAction_extractor(observations['pastAction'])
        features = [state_features, target_features, action_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class ActionStateTargetImageMaskNoiseExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
               and ("target" in obs_keys) \
               and ("pastAction" in obs_keys) \
               and ("noise_target" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys or "mask" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionStateTargetImageMaskNoiseExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _noise_target_features_dim = set_mlp_feature_extractor(self, "noise_target", observation_space["noise_target"], net_arch.get("noise_target", {}), activation_fn)
        # Convolution layers for image/mask observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key or "mask" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim + _target_features_dim + _action_features_dim + _noise_target_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        target_features = self.target_extractor(observations['target'])
        action_features = self.pastAction_extractor(observations['pastAction'])
        noise_target_features = self.noise_target_extractor(observations['noise_target'])
        features = [state_features, target_features, action_features, noise_target_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class SwarmStateTargetImageExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
               and ("target" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys) \
               and ("swarm" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(SwarmStateTargetImageExtractor, self).__init__(observation_space=observation_space,
                                                             features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        _swarm_features_dim = set_mlp_feature_extractor(self, "swarm", observation_space["swarm"], net_arch.get("state", {}), activation_fn) * \
                              observation_space["swarm"].shape[0]

        # Convolution layers for image/mask observations
        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))

        self._features_dim = _state_features_dim + _target_features_dim + _swarm_features_dim + sum(_image_features_dims)

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        target_features = self.target_extractor(observations['target'])
        swarm_features = []
        for i in range(observations['swarm'].shape[1]):
            swarm_features.append(self.swarm_extractor(observations['swarm'][:, i, :]))
        # swarm_features = [self.state_extractor(agent) for agent in observations['swarm']]
        features = [state_features, target_features] + swarm_features
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge state/target features and image features
        return th.cat(features, dim=1)


class StateIndexExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        # for key in observation_space.keys():
        #     assert key in "state" in key or "target" in key
        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) and ("index" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateIndexExtractor, self).__init__(observation_space=observation_space,
                                                   features_dim=self.features_dim)

        state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # pastAction_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        
        # self._features_dim = state_features_dim + index_features_dim + pastAction_features_dim
        self._features_dim = state_features_dim + index_features_dim 

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        # features = [state_features, index_features, action_features]
        features = [state_features, index_features]
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class ActionStateImageExtractor(CustomBaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
               and ("pastAction" in obs_keys) \
               and ("index" in obs_keys) \
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(ActionStateImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim  + _action_features_dim + _index_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [state_features, action_features, index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class StateIndexActionExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        # for key in observation_space.keys():
        #     assert key in "state" in key or "target" in key
        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) and ("index" in obs_keys) and ("pastAction" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateIndexActionExtractor, self).__init__(observation_space=observation_space,
                                                   features_dim=self.features_dim)

        state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        pastAction_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        
        self._features_dim = state_features_dim + index_features_dim + pastAction_features_dim
        # self._features_dim = state_features_dim + index_features_dim 

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [state_features, index_features, action_features]
        # features = [state_features, index_features]
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class StateIndexImageExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
                and ("index" in obs_keys)\
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateIndexImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim   + _index_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [state_features,  index_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class StateIndexVdImageExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
                and ("index" in obs_keys)\
                and ("vd" in obs_keys)\
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateIndexVdImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _vd_features_dim = set_mlp_feature_extractor(self, "vd", observation_space["vd"], net_arch.get("vd", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim  + _vd_features_dim + _index_features_dim + sum(_image_features_dims)
        
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        vd_features = self.vd_extractor(observations['vd'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [state_features,  index_features, vd_features]
        for name in self._image_extractor_names:
            # image forward process
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)
        
class StateVdImageExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
                and ("vd" in obs_keys)\
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateVdImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        _vd_features_dim = set_mlp_feature_extractor(self, "vd", observation_space["vd"], net_arch.get("vd", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        # _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        
        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim  + _vd_features_dim + sum(_image_features_dims)
        
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        vd_features = self.vd_extractor(observations['vd'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        # index_features = self.index_extractor(observations['index'])
        features = [state_features, vd_features]
        for name in self._image_extractor_names:
            # image forward process
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class NoiseStateIndexImageExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):

        obs_keys = list(observation_space.keys())
        assert  ("noise_state"  in obs_keys) \
                and ("index" in obs_keys)\
               and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(NoiseStateIndexImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        # _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        _noise_state_features_dim = set_mlp_feature_extractor(self, "noise_state", observation_space["noise_state"], net_arch.get("noise_state", {}), activation_fn)
        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim =  _noise_state_features_dim + _index_features_dim + sum(_image_features_dims)
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        # state_features = self.state_extractor(observations['state'])
        noise_state_features = self.noise_state_extractor(observations['noise_state'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        index_features = self.index_extractor(observations['index'])
        features = [index_features , noise_state_features]
        for name in self._image_extractor_names:
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class StateLatentExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        # for key in observation_space.keys():
        #     assert key in "state" in key or "target" in key
        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateLatentExtractor, self).__init__(observation_space=observation_space,
                                                   features_dim=self.features_dim)

        state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # pastAction_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        # index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        
        self._features_dim = state_features_dim
        # self._features_dim = state_features_dim + index_features_dim 

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        # index_features = self.index_extractor(observations['index'])
        features = [state_features]
        # features = [state_features, index_features]
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)

class StateIndexVdExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 net_arch: Dict = {},
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        # for key in observation_space.keys():
        #     assert key in "state" in key or "target" in key
        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) and ("index" in obs_keys)and ("vd" in obs_keys) 

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateIndexVdExtractor, self).__init__(observation_space=observation_space,
                                                   features_dim=self.features_dim)

        state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # pastAction_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        vd_features_dim = set_mlp_feature_extractor(self, "vd", observation_space["vd"], net_arch.get("vd", {}), activation_fn)
        index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        
        self._features_dim = state_features_dim + index_features_dim + vd_features_dim
        # self._features_dim = state_features_dim + index_features_dim 
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True
            
    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        vd_features = self.vd_extractor(observations['vd'])
        index_features = self.index_extractor(observations['index'])
        features = [state_features, index_features, vd_features]
        # features = [state_features, index_features]
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)


class StateImageExtractor(BaseFeaturesExtractor):
    backbone_alias: Dict = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_l": models.efficientnet_v2_l,
        "efficientnet_m": models.efficientnet_v2_m,
        "efficientnet_s": models.efficientnet_v2_s,
        "mobilenet_l": models.mobilenet_v3_large,
        "mobilenet_s": models.mobilenet_v3_small,
    }

    def __init__(self,
                observation_space: spaces.Dict,
                net_arch: Dict = {},
                activation_fn: Type[nn.Module] = nn.ReLU,
                ):

        obs_keys = list(observation_space.keys())
        assert ("state" in obs_keys) \
            and ("image" in obs_keys or "color" in obs_keys or "depth" in obs_keys)

        # Default feature dimension is 1
        self._features_dim = 1
        super(StateImageExtractor, self).__init__(observation_space=observation_space,
                                                        features_dim=self.features_dim)

        _state_features_dim = set_mlp_feature_extractor(self, "state", observation_space["state"], net_arch.get("state", {}), activation_fn)
        # _vd_features_dim = set_mlp_feature_extractor(self, "vd", observation_space["vd"], net_arch.get("vd", {}), activation_fn)
        # _target_features_dim = set_mlp_feature_extractor(self, "target", observation_space["target"], net_arch.get("target", {}), activation_fn)
        # _action_features_dim = set_mlp_feature_extractor(self, "pastAction", observation_space["pastAction"], net_arch.get("pastAction", {}), activation_fn)
        # _index_features_dim = set_mlp_feature_extractor(self, "index", observation_space["index"], net_arch.get("index", {}), activation_fn)
        
        # Convolution layers for image observations

        _image_features_dims = []
        self._image_extractor_names = []
        for key in observation_space.keys():
            if "image" in key or "color" in key or "depth" in key:
                _image_features_dims.append(set_cnn_feature_extractor(self, key, observation_space[key], net_arch.get(key, {}), activation_fn))
        self._features_dim = _state_features_dim + sum(_image_features_dims)
        
        if net_arch.get("recurrent", None) is not None:
            _hidden_features_dim = set_recurrent_feature_extractor(self, self._features_dim, net_arch.get("recurrent"))
            self._features_dim = _hidden_features_dim
            self._is_recurrent = True

    def forward(self, observations):
        state_features = self.state_extractor(observations['state'])
        # vd_features = self.vd_extractor(observations['vd'])
        # target_features = self.target_extractor(observations['target'])
        # action_features = self.pastAction_extractor(observations['pastAction'])
        # index_features = self.index_extractor(observations['index'])
        features = [state_features]
        for name in self._image_extractor_names:
            # image forward process
            x = getattr(self, name)(observations[name.split("_")[0]])
            features.append(x)
        # Merge all features
        if hasattr(self, "recurrent_extractor"):
            features, h = self.recurrent_extractor(th.cat(features, dim=1).unsqueeze(0), observations['latent'].unsqueeze(0))
            return features[0], h[0]
        else:
            return th.cat(features, dim=1)
        
def debug():
    test = 1


if __name__ == "__main__":
    debug()
