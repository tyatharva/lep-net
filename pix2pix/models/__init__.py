"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from models.base_model import BaseModel
import torch
import torch.nn as nn
from pytorch_msssim import ssim, SSIM


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance


class WeightedL2Loss(nn.Module):
    def __init__(self, weight_zero=1.0, weight_precip=10.0):
        super(WeightedL2Loss, self).__init__()
        self.weight_zero = weight_zero  # Weight for dry areas (pixels with 0 precipitation)
        self.weight_precip = weight_precip  # Weight for wet areas (pixels with precipitation)

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        weight_map = torch.where(target == 0, self.weight_zero, self.weight_precip)
        weighted_loss = loss * weight_map
        return weighted_loss.mean()


class WeightedL1Loss(nn.Module):
    def __init__(self, threshold=2.0, alpha=10.0):
        super(WeightedL1Loss, self).__init__()
        self.threshold = threshold
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        l1_loss = torch.abs(y_pred - y_true)
        weights = 1 + self.alpha * torch.relu(y_true - self.threshold)
        weighted_loss = weights * l1_loss
        return weighted_loss.mean()


class SSIML2Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(SSIML2Loss, self).__init__()
        self.alpha = alpha  # Balance between SSIM and L2
        self.l2_loss = nn.MSELoss()

    def forward(self, pred, target):
        ssim_val = ssim(pred, target, data_range=1, size_average=True)
        l2_val = self.l2_loss(pred, target)
        return (1 - self.alpha) * (1 - ssim_val) + self.alpha * l2_val


class CustomHybridLoss(nn.Module):
    def __init__(self, alpha=0.5, weight_zero=1.0, weight_precip=10.0):
        super(CustomHybridLoss, self).__init__()
        self.alpha = alpha
        self.weighted_mse = WeightedL2Loss(weight_zero, weight_precip)
        
    def forward(self, pred, target):
        l1_loss = torch.abs(pred - target).mean()  # L1 for intensity accuracy
        ssim_loss = 1 - ssim(pred, target, data_range=1, size_average=True)
        weighted_mse_loss = self.weighted_mse(pred, target)  # MSE focusing on precip areas
        
        # Combine the losses
        total_loss = (self.alpha * l1_loss) + (1 - self.alpha) * ssim_loss + weighted_mse_loss
        return total_loss


class LossFunctionSelector:
    def __init__(self):
        self.loss_functions = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
            'huber': nn.HuberLoss(),
            'wl1': WeightedL1Loss(),
            'wl2': WeightedL2Loss(),
            'ssiml2': SSIML2Loss(), 
            'hybrid': CustomHybridLoss()
        }

    def get_loss_function(self, loss_type):
        """
        Returns the corresponding loss function based on the input string.
        
        Args:
            loss_type (str): The type of loss function ('l1' or 'l2').
        
        Returns:
            torch.nn.Module: The corresponding loss function.
        
        Raises:
            ValueError: If the loss type is not recognized.
        """
        if loss_type.lower() in self.loss_functions:
            return self.loss_functions[loss_type.lower()]
        else:
            raise ValueError(f"Loss type '{loss_type}' is not recognized.")
