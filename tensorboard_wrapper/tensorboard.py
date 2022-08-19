from collections import defaultdict
from datetime import datetime
import os
import shutil

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from .exceptions import BoardAlreadyExistsException


class Tensorboard():
    def __init__(self, name=None, path=None, delete=False):
        """
        Params:
            name: Used for naming the run. 
            If no name is passed, the run will be saved using the current time.
            Default: None

            path: Defines where the log file should be saved. 
            If no path is passed, the run will be saved at './runs/'.
            Default: None

            delete: Whether the tensorboard should delete a run in the same path with the same name.
            If there is a log file a BoardAlreadyExistsException will be raised.
            Default: False
        """
        path = './runs/' if path is None else path
        path = f'{path}/' if '/' != path[-1] else path

        if os.path.exists(path) and delete:
            shutil.rmtree(f'{path}')
        elif os.path.exists(path) and not delete:
            raise BoardAlreadyExistsException(path)
        else:
            os.makedirs(path, exist_ok=True)

        if name is None:
            name = datetime.now()
            name = name.strftime("%m-%d-%Y %H.%M.%S")
            self.writer = SummaryWriter(f'{path}{name}')
        else:
            self.writer = SummaryWriter(f'{path}{name}')

        self.epoch = defaultdict(int)
        self.histograms = defaultdict(list)

    def add_graph(self, model, data):
        """
        """
        self.writer.add_graph(model, data)

    def add_grid(self, prior, epoch=None, **kwargs):
        """
        Add image data to summary, by transforming the kwargs param into a grid.

        Params:
            prior: Name it will used to create the divisions in Tensorboard.

            epoch: Wich epoch this grid belongs.
            It should be a string.
            Defualt: None.

            **kwargs:the key, value of the entry into the Tensorboard.
        """
        if not isinstance(epoch, str) and epoch is not None:
            raise Exception('Tensorboard: epoch should be a str or None')

        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        for i in kwargs:
            grid = torchvision.utils.make_grid(kwargs[i], normalize=True)
            self.writer.add_image(f'{prior}/{i}', grid, epoch)

    def add_histogram(self, histogram, epoch=None):
        """
        Add histogram to summary. 
        This method adds once per epoch. The results will only show after a step call.

        Params
            epoch: Key to storage values in.

            histogram: Value it should store.
        """
        if not isinstance(histogram, torch.Tensor):
            raise Exception('Tensorboard: histogram should be a Tensor')
        if not isinstance(epoch, str) and epoch is not None:
            raise Exception('Tensorboard: epoch should be a str or None')            
        
        if histogram.is_cuda:
            histogram = histogram.cpu()
        histogram = histogram.tolist()

        epoch = 'default' if epoch is None else epoch
        self.histograms[epoch].extend(histogram)
        
    def add_hparams(self, hparams, metrics={}):
        """
        Add a set of hyperparameters to summary.

        Params:
            params: Hyperparameters used.

            metrics: Eventual metrics.
            Default: Empty dict {}
        """
        if hparams is None or not isinstance(hparams, dict):
            raise Exception('Tensorboard: hparams should be a dictionary.')
        if not isinstance(metrics, dict):
            raise Exception('Tensorboard: metrics should be a dictionary.')
        self.writer.add_hparams(hparams, metrics)

    def add_image(self, title, image, prior=None, epoch=None):
        """
        Add a single image data to summary.

        Params:
            prior: Name it will used to create the divisions in Tensorboard.

            title: Name it will appear in the Tensorboard.

            image: Image it will save.

            epoch: Wich epoch this grid belongs.
            It should be a string.
            Defualt: None.
        """
        if not isinstance(prior, str) and prior is not None:
            raise Exception('Tensorboard: prior should be a string or None.')
        if not isinstance(title, str):
            raise Exception('Tensorboard: title should be a string.')
        if not isinstance(image, (torch.Tensor, np.ndarray)):
            raise Exception('Tensorboard: title should be a Tensor or a Numpy Array.')

        if len(image.shape) > 3:
            raise Exception('Tensorboard: for more than one image use "add_grid"')
        elif image.shape[0] not in [1, 3]:
            raise Exception('Tensorboard: image should be of shape CxHxW.')

        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        name = f'{prior}/{title}' if prior is not None else title
        self.writer.add_image(name, image, epoch)

    def add_scalar(self, title, value, epoch=None, prior=None):
        """
        Add a single scalar data to summary.

        Params:
            prior: Name it will used to create the divisions in Tensorboard.

            title: Name it will appear in the Tensorboard.

            image: Image it will save.

            epoch: Wich epoch this grid belongs.
            It should be a string.
            Defualt: None.
        """
        if not isinstance(title, str):
            raise Exception('Tensorboard: title should be a string.')
        if not isinstance(value, (int, float, torch.Tensor)):
            raise Exception('Tensorboard: value should be an int, float or torch.Tensor.')
        if not isinstance(epoch, str) and epoch is not None:
            raise Exception('Tensorboard: epoch should be a string or None.')
        if not isinstance(prior, str) and prior is not None:
            raise Exception('Tensorboard: prior should be a string or None.')

        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        name = f'{prior}/{title}' if prior is not None else title
        self.writer.add_scalar(name, value, epoch)

    def add_scalars(self, epoch=None, prior=None, **kwargs):
        """
        Add scalars data to summary.

        Params:
            prior: Name it will used to create the divisions in Tensorboard.

            image: Image it will save.

            epoch: Wich epoch this grid belongs.
            It should be a string.
            Defualt: None.

            **kwargs: Keys and values of all entries into the Tensorboard.
        """
        if not isinstance(kwargs, dict):
            raise Exception('Tensorboard: kwargs should be a dict.')
        if not isinstance(epoch, str) and epoch is not None:
            raise Exception('Tensorboard: epoch should be a string or None.')
        if not isinstance(prior, str) and prior is not None:
            raise Exception('Tensorboard: prior should be a string or None.')
        
        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        for i in kwargs:
            if not isinstance(kwargs[i], (int, float, torch.Tensor)):
                raise Exception('Tensorboard: value should be an int, float or torch.Tensor.')
            name = f'{prior}/{i}' if prior is not None else i
            self.writer.add_scalar(name, kwargs[i], epoch)
    
    def close(self):
        """
        Makes sure that all entries were written into the log file.
        """
        self.writer.flush()
        self.writer.close()

    def step(self, epoch=None):
        """
        Step a certain epoch or all epochs if no param is given.

        Param:
            epoch: Could be a string, a list of string, or None.
            Default: None
        """
        if not isinstance(epoch, (str, list)) and epoch is not None:
            raise Exception('Tensorboard: epoch should be a string, a list of strings or None.')
        if isinstance(epoch, list) and not all(key in self.epoch.keys() for key in epoch):
            raise Exception('Tensorboard: one of the epochs specified does not exist.')
        if not isinstance(epoch, list) and epoch not in self.epoch.keys() and epoch is not None:
            raise Exception('Tensorboard: the epoch specified does not exist.')  

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
        """
        Add all histograms to summary at the end of an epoch.
        """
        for title in self.histograms:
            self.writer.add_histogram(title, np.array(self.histograms[title]), self.epoch[epoch])

