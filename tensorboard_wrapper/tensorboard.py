from collections import defaultdict
from datetime import datetime
import os
import shutil

import numpy
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
        elif os.path.exists(path)and delete:
            raise BoardAlreadyExistsException(path)
        else:
            os.makedirs(path)

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

    def add_image(self, prior, title, image, epoch=None):
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
        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        self.writer.add_image(f'{prior}/{title}', image, epoch)

    def add_scalar(self, prior, title, value, epoch=None):
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
        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        self.writer.add_scalar(f'{prior}/{title}', value, epoch)

    def add_scalars(self, prior, epoch=None, **kwargs):
        """
        Add scalars data to summary.

        Params:
            prior: Name it will used to create the divisions in Tensorboard.

            image: Image it will save.

            epoch: Wich epoch this grid belongs.
            It should be a string.
            Defualt: None.

            **kwargs:the key, value of the entry into the Tensorboard.
        """
        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        for i in kwargs:
            self.writer.add_scalar(f'{prior}/{i}', kwargs[i], epoch)
    
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
        if self.histograms[epoch]:
            self.__save_histogram(epoch)
            self.histograms = defaultdict(list)

        if epoch is not None:
            self.epoch[epoch] += 1
        elif isinstance(epoch):
            for key in epoch:
                self.epoch[key] += 1
        elif len(self.epoch.keys()) > 0:
            for key in self.epoch.keys():
                self.epoch[key] += 1
        else:
            self.epoch['default'] += 1

    def __save_histogram(self, epoch):
        """
        Add all histograms to summary at the end of an epoch.
        """
        for title in self.histograms:
            self.writer.add_histogram(title, numpy.array(self.histograms[title]), self.epoch[epoch])

