from collections import defaultdict
import os
import shutil
import unittest

import torch
from torch.utils.tensorboard import SummaryWriter

from Tensorboard import Tensorboard
from Exceptions import BoardAlreadyExistsException

class TestCases(unittest.TestCase):

    def get_path(self):
        return f'{self.path}/{self.name}'

    def prior_positive_test(self):
        if os.path.exists(self.get_path()):
            shutil.rmtree(self.path)
    
    def prior_negative_test(self):
        if not os.path.exists(self.get_path()):
            os.makedirs(self.get_path())

    def create_board(self) -> Tensorboard:
        return Tensorboard(
            name=self.name,
            path=self.path,
            delete=False
        )

    def test_init(self):
        '''
        '''
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()

        board = Tensorboard(
            name=self.name,
            path=self.path,
            delete=False,
        )

        try:
            assert isinstance(board.writer, SummaryWriter)
            assert board.writer.log_dir == self.get_path()
            assert os.path.exists(self.get_path())
            assert isinstance(board.epoch, defaultdict)
            assert isinstance(board.histograms, defaultdict)
        finally:
            shutil.rmtree(self.path)

    def test_init_already_exists(self):
        '''
        '''

        self.name = 'Test'
        self.path = './Test'
        self.prior_negative_test()
        
        with self.assertRaises(BoardAlreadyExistsException) as context:
            Tensorboard(
                name=self.name,
                path=self.path,
                delete=False
            )

        message = 'Tensorboard: ./Test/ already exists'
        self.assertIn(message, str(context.exception))

    def test_add_grid(self):
        '''
        '''
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()

        board = self.create_board()

        images = torch.Tensor(size=(32, 3, 224, 224))
        board.add_grid(
            prior='Test',
            epoch=None,
            images=images,
        )

        board.add_grid(
            prior='Test',
            epoch='Test',
            images=images,
        )

    def test_add_grid_negative_epoch(self):
        '''
        '''
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()

        board = self.create_board()

        images = torch.Tensor(size=(32, 3, 224, 224))        
        with self.assertRaises(TypeError) as context:
            board.add_grid(
                epoch=[],
                images=images,
            )

        with self.assertRaises(Exception) as context:
            board.add_grid(
                prior='Test',
                epoch=[],
                images=images,
            )
        self.assertIn(
            'Tensorboard: epoch should be a str or None',
            str(context.exception)
        )
    
    def test_histogram(self):
        '''
        '''
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()        

        board = self.create_board()

        hist = torch.Tensor(size=(7, 7)).flatten()
        board.add_histogram(
            epoch='Test',
            histogram=hist,
        )

        assert board.histograms['Test'] == hist.tolist()
        assert len(board.histograms.keys()) == 1

        board.add_histogram(
            histogram=hist
        )

        assert board.histograms['default'] == hist.tolist()
        assert len(board.histograms.keys()) == 2

    def test_histogram_negative(self):
        '''
        '''
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        board = self.create_board()

        hist = torch.Tensor(size=(7, 7)).flatten()
        with self.assertRaises(TypeError) as context:
            board.add_histogram(
                epoch=[],
            )

        with self.assertRaises(Exception) as context:
            board.add_histogram(
                epoch=[],
                histogram=hist,
            )
        self.assertIn(
            'Tensorboard: epoch should be a str or None',
            str(context.exception)
        )
        assert len(board.histograms.keys()) == 0

        with self.assertRaises(Exception) as context:
            board.add_histogram(
                histogram=hist.tolist(),
            )
        
        self.assertIn(
            'Tensorboard: histogram should be a Tensor',
            str(context.exception)
        )
        assert len(board.histograms.keys()) == 0

    def test_hparams(self):
        '''
        '''
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        board = self.create_board()  
        board.add_hparams({}, {})
        board.add_hparams({})

    def test_hparams_negative(self):
        '''
        '''
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        board = self.create_board()
        
        with self.assertRaises(Exception) as context:
            board.add_hparams(hparams=[])
        self.assertIn(
            'Tensorboard: hparams should be a dictionary.',
            str(context.exception)
        )

        with self.assertRaises(TypeError) as context:
            board.add_hparams(metrics={})

        with self.assertRaises(Exception) as context:
            board.add_hparams(hparams={}, metrics=[])
        self.assertIn(
            'Tensorboard: metrics should be a dictionary.',
            str(context.exception)
        )

if __name__ == "__main__":
    unittest.main()
    # TestCases().test_add_grid()