from .initializer import Initializer

'''
Logic for training and evaluation processes.
'''
class Processor(Initializer):

    def train(self):

        return None
    
    def eval(self):

        return None
    
    def start(self):
        '''
        Is the main entry point for the processor and orchestrates the overall workflow.
        Manages the full training or evaluation pipeline.
        '''
        
        return None
    
    def extract(self):
        '''
        Processes only one batch of data and focuses on data extraction and saving.
        Is designed to extract and save model outputs,features, weights, and related data.
        It loads a trained model checkpoint, processes the data through the model in evaluation mode, 
        applies softmax to outputs, and saves everything to a .npz file. 
        This is typically used post-training to generate data for plotting or inspecting model behavior, 
        rather than for training or standard evaluation.
        '''

        return None