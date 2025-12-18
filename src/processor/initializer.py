from feeder.itwpolimi_feeder import ITWPOLIMI_Feeder

class Initializer():
    """
        Initializes all static components of the project: configuration,
        data loaders, models, optimizer, and loss function.
    """
    
    def __init__(self, config):
        '''
        Load static parameters like embedding sizes, joint numbers, T size, etc.
        '''
        self.config = config

    def init_dataloader(self):
        '''
        Initialize the dataloader based on the configuration.
        '''
        
        return None
    
    def init_model(self):
        '''
        Initialize the model based on the configuration.
        '''
        
        return None
    
    def load_checkpoint(self):
        '''
        Load the model checkpoint from the specified filename.
        '''
        
        return None
    
    def init_loss(self):
        '''
        Initialize the loss function based on the configuration.
        '''
        
        return None
    
    def init_optimizer(self):
        '''
        Initialize the optimizer based on the configuration.
        '''

        return None