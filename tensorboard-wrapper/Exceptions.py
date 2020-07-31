class BoardAlreadyExistsException(Exception):
    '''
    Exception used for when the Tensorboard class cannot delete a folder
    that already exists. This exception should not be use for anything else.
    '''
    def __init__(self, name):
        super().__init__(f'Tensorboard: {name} already exists')