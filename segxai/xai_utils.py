from socket import if_indextoname
import numpy as np
import torch
from typing import List
import functools
# https://github.com/linhaoqi027/SEG-GRAD-CAM/blob/master/gradcam_unet.py


class FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def get_activation(self, name):
        def hook(model, inp, output):
            self.target_activation = inp

        return hook

    def __call__(self, x):

        # register activation
        # for i, layer in enumerate(list(self.model._modules.keys())):
        #     if layer == self.target_layer:
        #         getattr(self.model, layer).register_forward_hook(
        #             self.get_activation(layer))

        getattr(self.model, self.target_layer).register_forward_hook(
            self.get_activation(self.target_layer)
        )

        # perform forward pass
        out = self.model(x)

        return self.target_activation, out


class ModelOutputs:
    """Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers.
    returns target activations and model output (logits)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layer)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):

        # performs forward pass when called
        target_activations, x = self.feature_extractor(x)

        return target_activations, x

def normalize(x: np.array) -> np.array:
    return (x - x.min()) / (x.max() - x.min())




def logit2pred(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return x.argmax(dim)


# https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/arraysetops.py#L138-L317
def _unpack_tuple(x):
    """Unpacks one-element tuples for use as return values"""
    if len(x) == 1:
        return x[0]
    else:
        return x


def iou_positives(
    pred: torch.Tensor, mask: torch.Tensor, labels: list, fval=np.nan
) -> np.array:
    iou = np.full(len(labels), fill_value=fval)
    for c in labels:
        # only positive predicitons
        if c in pred:
            inter = ((pred == c) & (mask == c)).sum().item()
            union = ((pred == c) | (mask == c)).sum().item()
            iou[c] = inter / union

    return iou


def torch_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x (numpy array).
    default : C x H x W => C x H x W
    """

    x_copy = x - torch.amax(x, axis=axis)
    y = torch.exp(x_copy)
    return y / y.sum(axis=axis)


def sequpsamp(gradients:np.array, feature_maps:List[np.array]):
    from skimage.transform import rescale

    # for the last layer simple upsampling is performed
    for step in range(len(feature_maps)-1):
        gradients = rescale(gradients, 2) * feature_maps[len(feature_maps) - 2 - step]
        
    ret = rescale(gradients, 2)

    return ret


def collaps(src: np.array, grds: np.array, method="mean") -> np.array:
    assert (
        src.shape == grds.shape
    ), f"Source has to have same shape as target gradiets, but {src.shape} and {grds.shape} was given!"
    valid_methods = ["mean", "hadamard"]
    assert method in valid_methods, f"Chosen method not in {valid_methods}"

    if method == "mean":
        # CxHxW -> C
        weights = np.mean(grds, axis=(1, 2))

        # weighted sum
        # CxHxW -> HxW
        return np.einsum("k,kli->li", weights, src)

    if method == "hadamard":
        return (src * grds).sum(0)
