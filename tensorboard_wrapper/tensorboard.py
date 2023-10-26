"""Module for Tensorboard class."""
from typing import Union, List, Any, Dict
from numbers import Number
from collections import defaultdict
from datetime import datetime
import os
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from .exceptions import BoardAlreadyExistsException


class Tensorboard():
    """Tensorboard class which wraps tensorboard own class and adds some utility functions."""

    def __init__(
        self,
        name: str = datetime.now().strftime("%m-%d-%Y %H.%M.%S"),
        path: str = "./runs/",
        delete: bool = False
    ) -> None:
        """
        Params:
            name (str): Used for naming the run. If no name is passed, the run will
                be saved using the current time. Defaults to datetime.now().
            path (str): Defines where the log file should be saved. If no path is
                passed, the run will be saved at './runs/'. Defaults to './runs/'.
            delete (bool): Whether the tensorboard should delete a run in the same
                path with the same name. Defaults to False.

        Raises:
            BoardAlreadyExistsException: if delete is True and a board if there is
                a board with the same name in the same path.
        """
        path = f'{path}/' if '/' != path[-1] else path

        if os.path.exists(f"{path}{name}") and delete:
            shutil.rmtree(f"{path}{name}")
        elif os.path.exists(f"{path}{name}") and not delete:
            raise BoardAlreadyExistsException(path)
        else:
            os.makedirs(path, exist_ok=True)

        self.writer = SummaryWriter(f'{path}{name}')
        self.epoch = defaultdict(int)
        self.histograms = defaultdict(list)

    def add_graph(self, model: nn.Module, data: torch.Tensor) -> None:
        """Adds a graph to the tensorboard.

        Args:
            model (nn.Module): Pytorch model.
            data (torch.Tensor): PyTorch tensor.
        """
        self.writer.add_graph(model, data)

    def add_grid(self, prior: str, epoch: str = None, **kwargs) -> None:
        """Add image data to summary, by transforming the kwargs param into a grid.

        Params:
            prior (str): Name it will used to create the divisions in Tensorboard.
            epoch (str): Wich epoch this grid belongs. Defualts to None.
            **kwargs (Dict[str, Any]): the key, value of the entry into the Tensorboard.

        Raises:
            ValueError: if epoch is not a string or is not None.
        """
        if not isinstance(epoch, str) and epoch is not None:
            raise ValueError('Tensorboard: epoch should be a str or None')

        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        for key, value in kwargs.items():
            grid = torchvision.utils.make_grid(value, normalize=True)
            self.writer.add_image(f'{prior}/{key}', grid, epoch)

    def add_histogram(self, histogram: torch.Tensor, epoch: str = None) -> None:
        """Add histogram to summary. This method adds once per epoch. The results
            will only show after a step call.

        Args:
            histogram (torch.Tensor): Value it should store.
            epoch (str): Key to storage values in. Defaults to None.

        Raises:
            ValueError: if histogram is not a torch.Tensor.
            ValueError: if epoch is not a str or None.
        """
        if not isinstance(histogram, torch.Tensor):
            raise ValueError('Tensorboard: histogram should be a Tensor')

        if not isinstance(epoch, str) and epoch is not None:
            raise ValueError('Tensorboard: epoch should be a str or None')

        if histogram.is_cuda:
            histogram = histogram.cpu()
        histogram = histogram.tolist()

        epoch = 'default' if epoch is None else epoch
        self.histograms[epoch].extend(histogram)

    def add_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, Any] = None):
        """Add a set of hyperparameters to summary.

        Args:
            params (Dict[str, Any]): Hyperparameters used.
            metrics (Dict[str, Any]): Eventual metrics. Default: Empty dict {}

        Raises:
            ValueError: if hparams is not a dictionary or None.
            ValueError: if metrics is not a Hashmap.
        """
        if metrics is None:
            metrics = {}

        if hparams is None or not isinstance(hparams, dict):
            raise ValueError('Tensorboard: hparams should be a dictionary.')

        if not isinstance(metrics, dict):
            raise ValueError('Tensorboard: metrics should be a dictionary.')

        self.writer.add_hparams(hparams, metrics)

    def add_image(
        self,
        title: str,
        image: Union[torch.Tensor, np.ndarray],
        prior: str = None,
        epoch: str = None
    ) -> None:
        """Add a single image data to summary.

        Args:
            title (str): Name it will appear in the Tensorboard.
            image (Union[torch.Tensor, np.ndarray]): Image it will save.
            prior (str): Name it will used to create the divisions in Tensorboard.
            epoch (str): Wich epoch this grid belongs. Defualts to None.

        Raises:
            ValueError: if prior is not a str or None.
            ValueError: if title is not a str.
            ValueError: if image is not a torch.Tensor or np.ndarray.
            ValueError: if image is a 4 dimensional array.
            ValueError: if image is not CxHxW.
        """
        if not isinstance(prior, str) and prior is not None:
            raise ValueError('Tensorboard: prior should be a string or None.')

        if not isinstance(title, str):
            raise ValueError('Tensorboard: title should be a string.')

        if not isinstance(image, (torch.Tensor, np.ndarray)):
            raise ValueError('Tensorboard: title should be a Tensor or a Numpy Array.')

        if len(image.shape) > 3:
            raise ValueError('Tensorboard: for more than one image use "add_grid"')
        if image.shape[0] not in [1, 3]:
            raise ValueError('Tensorboard: image should be of shape CxHxW.')

        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        name = f'{prior}/{title}' if prior is not None else title
        self.writer.add_image(name, image, epoch)

    def add_scalar(
        self,
        title: str,
        value: Union[Number, torch.Tensor],
        prior: str = None,
        epoch: str = None,
    ) -> None:
        """Add a single scalar data to summary.

        Args:
            title (str): Name it will appear in the Tensorboard.
            value (Union[Number, torch.Tensor]): Scalar value it will save.
            prior (str): Name it will used to create the divisions in Tensorboard.
                Defaults to None.
            epoch (str): Wich epoch this grid belongs. Defualts to None.

        Raises:
            ValueError: if title is not a string.
            ValueError: if value is not a float, int or torch.Tensor.
            ValueError: if prior is not a string or None.
            ValueError: if epoch is not a string or None.
        """
        if not isinstance(title, str):
            raise ValueError('Tensorboard: title should be a string.')

        if not isinstance(value, (int, float, torch.Tensor)):
            raise ValueError('Tensorboard: value should be an int, float or torch.Tensor.')

        if not isinstance(epoch, str) and epoch is not None:
            raise ValueError('Tensorboard: epoch should be a string or None.')

        if not isinstance(prior, str) and prior is not None:
            raise ValueError('Tensorboard: prior should be a string or None.')

        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        name = f'{prior}/{title}' if prior is not None else title
        self.writer.add_scalar(name, value, epoch)

    def add_scalars(self, prior: str = None, epoch: str = None, **kwargs):
        """Add scalars data to summary.

        Args:
            prior (str): Name it will used to create the divisions in Tensorboard.
            epoch (str): Wich epoch this grid belongs. Defualts to None.
            **kwargs (Dict[str, Any]): Keys and values of all entries into the Tensorboard.

        Raises:
            ValueError: if kwargs is not a Dict[str, Any].
            ValueError: if epoch is not a string or None.
            ValueError: if prior is not a string or None.
        """
        if not isinstance(kwargs, dict):
            raise ValueError('Tensorboard: kwargs should be a dict.')

        if not isinstance(epoch, str) and epoch is not None:
            raise ValueError('Tensorboard: epoch should be a string or None.')

        if not isinstance(prior, str) and prior is not None:
            raise ValueError('Tensorboard: prior should be a string or None.')

        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        for key, value in kwargs.items():
            if not isinstance(value, (int, float, torch.Tensor)):
                raise ValueError('Tensorboard: value should be an int, float or torch.Tensor.')
            name = f'{prior}/{key}' if prior is not None else key
            self.writer.add_scalar(name, value, epoch)

    def close(self):
        """Makes sure that all entries were written into the log file."""
        self.writer.flush()
        self.writer.close()

    def step(self, epoch: Union[List[str], str] = None) -> None:
        """Step a certain epoch or all epochs if no param is given.

        Args:
            epoch (Uniont[List[str], str]): Could be a string, a list of string, or None.
                Defaults to None (advances all epochs).

        Raises:
            ValueError: if epoch is not a string or a list of strings.
            ValueError: if epoch is a list and one of them does not exists.
            ValueError: if the epoch does not exists.
        """
        if not isinstance(epoch, (str, list)) and epoch is not None:
            raise ValueError('Tensorboard: epoch should be a string, a list of strings or None.')

        if isinstance(epoch, list) and not all(key in self.epoch.keys() for key in epoch):
            raise ValueError('Tensorboard: one of the epochs specified does not exist.')

        if not isinstance(epoch, list) and epoch not in self.epoch.keys() and epoch is not None:
            raise ValueError('Tensorboard: the epoch specified does not exist.')

        if isinstance(epoch, list):
            for key in epoch:
                if self.histograms[key]:
                    self.__save_histogram(key)
                self.epoch[key] += 1
        elif epoch is not None:
            if self.histograms[epoch]:
                self.__save_histogram(epoch)
            self.epoch[epoch] += 1
        elif len(self.epoch.keys()) > 1:
            for key in self.epoch.keys():
                if self.histograms[key]:
                    self.__save_histogram(key)
                self.epoch[key] += 1
        else:
            if self.histograms['default']:
                self.__save_histogram('default')
            self.epoch['default'] += 1
        self.histograms = defaultdict(list)

    def __save_histogram(self, epoch):
        """Add all histograms to summary at the end of an epoch."""
        for title in self.histograms:
            self.writer.add_histogram(title, np.array(self.histograms[title]), self.epoch[epoch])
