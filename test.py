from collections import defaultdict
import os
import shutil
import unittest

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tensorboard_wrapper import Tensorboard
from tensorboard_wrapper import BoardAlreadyExistsException

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
            delete=True
        )

    def default(self) -> Tensorboard:        
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        return self.create_board()

    def test_init(self) -> None:
        """
        """
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
            assert isinstance(board.epoch['default'], int)
            assert isinstance(board.histograms, defaultdict)
            assert isinstance(board.histograms['default'], list)
        finally:
            board.close()
            shutil.rmtree(self.path)

    def test_init_already_exists(self):
        """
        """

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
        """
        """
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
        board.close()

    def test_add_grid_negative_epoch(self):
        """
        """
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
        
        board.close()
    
    def test_histogram(self):
        """
        """
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()        

        board = self.create_board()

        hist = torch.Tensor(size=(7, 7)).flatten()
        board.add_histogram(
            epoch='test',
            histogram=hist,
        )

        assert board.histograms['test'] == hist.tolist()
        assert len(board.histograms.keys()) == 1

        board.add_histogram(
            histogram=hist
        )

        assert board.histograms['default'] == hist.tolist()
        assert len(board.histograms.keys()) == 2
        
        board.close()

    def test_histogram_negative(self):
        """
        """
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
        
        board.close()

    def test_hparams_positive(self):
        """
        """
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        board = self.create_board()  
        board.add_hparams({}, {})
        board.add_hparams({})

        board.close()

    def test_hparams_negative(self):
        """
        """
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
        
        board.close()

    def test_add_image_positive(self):
        """
        """
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        board = self.create_board()

        # Test image as torch.Tensor
        image = torch.Tensor(size=(3, 224, 224))
        board.add_image(prior='Test', title='Test', image=image, epoch='Test')

        # Test image as np.ndarray
        image = np.ndarray(shape=(3, 224, 224))
        board.add_image(prior='Test', title='Test', image=image, epoch='Test')

        # Test prior None
        board.add_image(title='Test', image=image, epoch='Test')
        # Test epoch None
        board.add_image(prior='Test', title='Test', image=image)
        # Test prior and epoch None
        board.add_image(title='Test', image=image)
        
        board.close()

    def test_add_image_negative(self):
        """
        """
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        board = self.create_board()

        image = torch.Tensor(size=(3, 224, 224))

        # Test prior different type
        with self.assertRaises(Exception) as context:
            board.add_image(prior={}, title='Test', image=image)
        self.assertIn(
            'Tensorboard: prior should be a string or None.',
            str(context.exception)
        )

        # Test title different type
        with self.assertRaises(Exception) as context:
            board.add_image(title={}, image=image)
        self.assertIn(
            'Tensorboard: title should be a string.',
            str(context.exception)
        )

        # Test image as PIL
        pil_image = transforms.ToPILImage()(image)
        with self.assertRaises(Exception) as context:
            board.add_image(title='Test', image=pil_image)
        self.assertIn(
            'Tensorboard: title should be a Tensor or a Numpy Array.',
            str(context.exception)
        )

        # Test image with 2 channels
        two_channels = torch.Tensor(size=(2, 224, 224))
        with self.assertRaises(Exception) as context:
            board.add_image(title='Test', image=two_channels)
        self.assertIn(
            'Tensorboard: image should be of shape CxHxW.',
            str(context.exception)
        )

        # Test image wrong order of dims
        wrong_order = torch.Tensor(size=(224, 224, 3))
        with self.assertRaises(Exception) as context:
            board.add_image(title='Test', image=wrong_order)
        self.assertIn(
            'Tensorboard: image should be of shape CxHxW.',
            str(context.exception)
        )

        # Test image as batched images        
        batched_images = torch.Tensor(size=(32, 224, 224, 3))
        with self.assertRaises(Exception) as context:
            board.add_image(title='Test', image=batched_images)
        self.assertIn(
            'Tensorboard: for more than one image use "add_grid"',
            str(context.exception)
        )

        board.close()

    def test_add_scalar_positive(self):
        """
        """
        board = self.default()

        value = torch.tensor(23)
        # Test sending all parameters
        board.add_scalar(title='Test', value=value, epoch='Test', prior='Test')

        # Test sending only prior
        board.add_scalar(title='Test', value=value, prior='Test')

        # Test sending only epoch
        board.add_scalar(title='Test', value=value, epoch='Test')

        # Test sending only necessary parameters
        board.add_scalar(title='Test', value=value)

        # Test value as int
        board.add_scalar(title='Test', value=23)
        
        # Test value as float
        board.add_scalar(title='Test', value=.23)

        # Test value as torch.Tensor
        board.add_scalar(title='Test', value=torch.tensor(23))
        
        board.close()

    def test_add_scalar_negative(self):
        """
        """
        board = self.default()

        value = torch.tensor(23)
        # Test title different type
        with self.assertRaises(Exception) as context:
            board.add_scalar(title=23, value=value)
        self.assertIn(
            'Tensorboard: title should be a string.',
            str(context.exception)
        )

        # Test value different types
        with self.assertRaises(Exception) as context:
            board.add_scalar(title='Test', value='23')
        self.assertIn(
            'Tensorboard: value should be an int, float or torch.Tensor.',
            str(context.exception)
        )

        # Test prior different type
        with self.assertRaises(Exception) as context:
            board.add_scalar(title='Test', value=value, prior=23)
        self.assertIn(
            'Tensorboard: prior should be a string or None.',
            str(context.exception)
        )

        # Test epoch different type
        with self.assertRaises(Exception) as context:
            board.add_scalar(title='Test', value=value, epoch=23)
        self.assertIn(
            'Tensorboard: epoch should be a string or None.',
            str(context.exception)
        )
        
        board.close()

    def test_add_scalars_positive(self):
        """
        """
        board = self.default()

        values = defaultdict(int)
        values['value1'] = torch.tensor(23)
        values['value2'] = .23
        values['values3'] = 23

        # Test sending all parameters
        board.add_scalars(prior='Test', epoch='Test', Test=123)

        # Test sending only prior
        board.add_scalars(prior='Test', Test=123)

        # Test sending only epoch
        board.add_scalars(epoch='Test', Test=123)

        # Test sending only necessary parameters
        board.add_scalars(Test=123)

        # Test value as int
        board.add_scalars(**values)

        board.close()

    def test_add_scalars_negative(self):
        """
        """
        board = self.default()

        values = defaultdict(int)
        values['value1'] = torch.tensor(23)
        values['value2'] = '.23'
        values['values3'] = 23

        # Test kwargs different types
        with self.assertRaises(Exception) as context:
            board.add_scalars(**values)
        self.assertIn(
            'Tensorboard: value should be an int, float or torch.Tensor.',
            str(context.exception)
        )


        # Test prior different type
        with self.assertRaises(Exception) as context:
            board.add_scalars(prior=23, Test=123)
        self.assertIn(
            'Tensorboard: prior should be a string or None.',
            str(context.exception)
        )

        # Test epoch different type
        with self.assertRaises(Exception) as context:
            board.add_scalars(epoch=23, Test=123)
        self.assertIn(
            'Tensorboard: epoch should be a string or None.',
            str(context.exception)
        )
        
        board.close()

    def test_delta_save_positive(self):
        """
        """
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        board = Tensorboard(
            name=self.name, 
            path=self.path, 
            delete=False,
            delta=30,
        )

        # Test if the model should save
        assert board.should()

        # Test if the model should not save
        board.step()
        assert not board.should()

        # Test upper limit
        board.epoch['default'] += 29
        assert board.should()

        # Test specific epoch
        board.epoch['test'] = 0
        assert board.should(epoch='test')

    def test_delta_save_negative(self) -> None:
        """
        """
        self.name = 'Test'
        self.path = './Test'
        self.prior_positive_test()
        board = Tensorboard(
            name=self.name, 
            path=self.path, 
            delete=False,
            delta=30,
        )
        
        # Test non existent epoch
        # Test wrong type epoch

    def test_step_positive(self):
        """
        """
        board = self.default()
        
        # Test if step increases in value
        assert board.epoch['default'] == 0
        board.step()
        assert board.epoch['default'] == 1

        # Test if it increases only the specified epoch
        board.epoch['test'] = 0
        board.step('test')
        assert board.epoch['default'] == 1
        assert board.epoch['test'] == 1

        # Test if accepts a list
        board.epoch['test2'] = 0
        board.step(['test', 'test2'])
        assert board.epoch['test'] == 2
        assert board.epoch['test2'] == 1
        assert board.epoch['default'] == 1

        # Test if it updates all epochs
        board.step()
        assert board.epoch['test'] == 3
        assert board.epoch['test2'] == 2
        assert board.epoch['default'] == 2

        # Test if saves the histogram
        board.add_histogram(torch.tensor([1, 2, 3]))
        board.step()
        assert len(board.histograms.keys()) == 0

    def test_step_negative(self) -> None:
        """
        """
        board = self.default()
        # Test epoch not type
        with self.assertRaises(Exception) as context:
            board.step(epoch=23)
        self.assertIn(
            'Tensorboard: epoch should be a string, a list of strings or None.',
            str(context.exception)
        )

        # Test epoch not in keys (list)
        board = self.default()
        board.step()
        with self.assertRaises(Exception) as context:
            board.step(epoch=['default', 'test'])
        self.assertIn(
            'Tensorboard: one of the epochs specified does not exist.',
            str(context.exception)
        )

        # Test epoch not in keys (str)
        board = self.default()
        with self.assertRaises(Exception) as context:
            board.step(epoch='test')
        self.assertIn(
            'Tensorboard: the epoch specified does not exist.',
            str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()