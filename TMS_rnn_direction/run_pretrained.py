import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

class Temporal_Learning(nn.Module):
    '''
        model: RNN architecture used for training.  
               Acceptable entries are LSTM and GRU.
        input_size: The number of expected features in the input 
                    For instance if you predict the next sample 
                    by looking at the past 3 samples, 
                    then input_size would be 3 
        hidden_size: number of features in the hidden state.
        dropout: introduces a dropout layer on the outputs of 
                 each LSTM layer except the last layer, 
                 with dropout probability equal to dropout.
    '''
    def __init__(self, model, input_size, hidden_size, dropout):
        super(Temporal_Learning, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        if self.model == 'lstm':
            self.lstm1 = nn.LSTMCell(input_size, hidden_size)
            self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        elif self.model == 'gru':
            self.gru1 = nn.GRUCell(input_size, hidden_size)
            self.gru2 = nn.GRUCell(hidden_size, hidden_size)        
        else:
            raise ValueError("Acceptable entries for model are 'lstm' and "
                             "'gru' You entered: ", model)
        
        self.linear = nn.Linear(hidden_size, 1)  

'''
    Loads the pretrained networks. They were trained by scaling the input data
    with minmax.
'''
def load_model(model, parser):
    mode = "LogScaler" 
    try:
        model.load_state_dict(torch.load("TrainedModels/SLB_Proj_"+mode+"_"
                                            +parser+"_polarform.model"))
    except RuntimeError:
        print("Runtime Error!")
        print(("Saved model must have the same network architecture with"
               " the CopyModel.\nRe-train and save again or fix the" 
               " architecture of CopyModel."))
        exit(1) # stop execution with error