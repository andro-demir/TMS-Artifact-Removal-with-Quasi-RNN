from argparse import ArgumentParser, ArgumentTypeError
from human_data_parser import parser
from sklearn.preprocessing import MinMaxScaler
from math import log
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-model", type=str, help=("RNN architectures " 
                        "used for training. Acceptable entries are 'LSTM' "
                        "and 'GRU'."), required=True)
    parser.add_argument("-optimizer", type=str, help=("Choose the optimization"
                        " technique. Acceptable entries are 'L-BFGS' and "
                        "'Adam'"), required=True)
    parser.add_argument("-future", type=int, help=("This model predicts future"
                        " number of samples. Enter the number of samples you "
                        "would like to predict."), required=True)
    parser.add_argument("-scaler", type=str, help=("Scaling method for the "
                        "input data. Acceptable entries are 'minmax' and "
                        "'log'."), required=True)
    parser.add_argument("-intensity", type=int, help=("Enter the TMS intensity"
                        " level (MSO). Acceptable entries are 10, 20, 30, 40, "
                        "50, 60, 70, 80 and 0. O for taking all intensity "
                        "levels."), required=True)
    parser.add_argument("-channel", type=int, help=("Enter the channel number."
                        " Acceptable entries are 0, 1 , ... 62."), 
                        required=True)
    args = parser.parse_args()
    return args

'''
    Stops execution with Assertion error if the entries for args.parser are not 
    acceptable.
    If args in the command line are legal, returns args.
'''
def pass_legal_args():
    acceptable_MSO = list(range(0, 90, 10))
    acceptable_channel = list(range(0, 63, 1))
    acceptable_scalers = ['minmax', 'log']
    args = get_args()

    assert args.model.lower() == "lstm" or args.model.lower() == "gru", ("\n"
           "Acceptable entries for argument model are: 'lstm' and 'gru'\nYou"
           " entered: " + args.model)
    assert args.optimizer.lower() == 'l-bfgs' or \
           args.optimizer.lower() == 'adam', ("\nAcceptable entries for " 
           "optimizer are l-bfgs and adam. You entered: " + args.optimizer)
    assert args.future > 0, "Future must be a positive integer."
    assert args.intensity in acceptable_MSO, ("Acceptable entries for TMS "
           "intensity (MSO) are 10, 20, 30, 40, 50, 60, 70, 80.\nYou entered "
           + args.intensity)
    assert args.channel in acceptable_channel, ("Acceptable entries for the "
           "EEG channels are 0, 1, 2, 3, ... 62.\nYou entered " + args.channel)
    assert args.scaler in acceptable_scalers, ("Acceptable entries for the "
           "scaling method are 'minmax' and 'log'.\nYou entered " + args.scaler)
    return args

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
        input: tensor containing the features of the input sequence
               of shape (channel_size, seq. length) 
        output: tensor containing the output features (h_t) from the last layer
                of the LSTM, for each t.
                of shape (channel_size, seq. length)
        batch = input.size(0)
        input_t: of shape(batch, input_size)
        h_t: tensor containing the hidden state for t = layer_num
             of shape (batch, hidden_size)
        c_t: tensor containing the cell state for t = layer_num
             of shape (batch, hidden_size)
        future: this model predicts future number of samples.
    '''
    def forward(self, input, device, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, 
                          dtype=torch.double).to(device)
        c_t = torch.zeros(input.size(0), self.hidden_size, 
                          dtype=torch.double).to(device)
        h_t2 = torch.zeros(input.size(0), self.hidden_size, 
                           dtype=torch.double).to(device)
        c_t2 = torch.zeros(input.size(0), self.hidden_size, 
                           dtype=torch.double).to(device)
        
        for i in range(input.size(1)-self.input_size):
            input_t = input[:,i:(i+self.input_size)]
            if self.model == 'lstm':
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            elif self.model == 'gru':
                h_t = self.gru1(input_t, h_t)
                h_t2 = self.gru2(h_t, h_t2)
            
            output = self.linear(h_t2)
            outputs += [output]
        
        for i in range(future): # for predicting the future samples
            inputs = outputs[-self.input_size:] 
            for i, tensor in enumerate(inputs):
                tensor_list = tensor.cpu().numpy().tolist()
                flat_list = [item for sublist in tensor_list for item 
                                                          in sublist]
                inputs[i] = flat_list
            inputs = np.array(inputs)
            inputs = torch.t(torch.from_numpy(inputs))
            if self.model == 'lstm':
                h_t, c_t = self.lstm1(inputs.to(device), (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            elif self.model == 'gru':
                h_t = self.gru1(inputs.to(device), h_t)
                h_t2 = self.gru2(h_t, h_t2)
            
            output = self.linear(h_t2)
            outputs += [output]
        
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs  


'''
    Loads the pretrained networks. They were trained by scaling the input data
    with minmax.
'''
def load_model(network, mode, optimizer, scaler, intensity, channel):
    try:
        network.load_state_dict(torch.load("../TrainedModels/tmseeg_" + mode +
                                           "_" + optimizer + "_" + scaler + "_" 
                                           + str(intensity) + "_" + 
                                           str(channel) + "_.model"))
    except RuntimeError:
        print("Runtime Error!")
        print(("Saved model must have the same network architecture with"
               " the CopyModel.\nRe-train and save again or fix the" 
               " architecture of CopyModel."))
        exit(1) # stop execution with error

'''
    If in minmax mode, transforms input by scaling them to range (0,1) linearly
    Transforms each trial in the range 0-1 seperately  
'''
def minmax_scale(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(np.transpose(data))        
    return np.transpose(data_scaled)

'''
    If in Log Scaling mode, transforms input in 2 dimensions
    with a log function of base 12.
'''
def log_scale(data, log_base=12):
    # make sure all samples are positive
    inc = 1 + abs(np.amin(data)) 
    data += inc
    scaler = lambda t: log(t, log_base)
    scaler = np.vectorize(scaler)
    data_scaled = scaler(data)                   
    return data_scaled, inc

'''
    Converts the data that is log scaled back to the original scale.
'''
def inv_logscale(data, inc, log_base=12):
    data = np.power(log_base, data)
    data -= inc
    return data

'''
    Draws the results.
'''
def plot_results(actual_output, model_output, args):
    plt.plot(actual_output, 'r', label='Actual')
    plt.plot(model_output, 'b', label='Prediction')
    plt.title('TMS Artifact Preditiction MSO:%s ch:%s' %(args.intensity,
              args.channel), fontsize=30)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (Discrete)')
    plt.legend()
    plt.show()

def main():
    args = pass_legal_args()
    # Loads the TMS-EEG data of desired intensity and from desired channel
    dp = parser() # Initializes the class, loads TMS-EEG data
    if args.intensity != 0:
        dp.get_intensity(args.intensity) # Calls the get_intensity method
        dp.get_channel(args.channel)     # Calls the get_channel method
        # Model expects object type of double tensor, write as type 'float64'
        unscaled_data = np.transpose(dp.channel_data).astype('float64')
    else:
        unscaled_data = dp.get_all_intensities(args.channel).astype('float64')
    # Shuffles the rows of the dataset:
    np.random.shuffle(unscaled_data) # in-place operation

    # Scaling the data:
    if args.scaler.lower() == "log":
        data, inc = log_scale(unscaled_data)
    elif args.scaler.lower() == "minmax":
        data = minmax_scale(unscaled_data)
    
    # Loads the pre-trained model's parameters to the network architecture 
    input_size, hidden_size, dropout = 5, 64, 0.5
    network = Temporal_Learning(args.model, input_size, hidden_size, dropout)
    load_model(network, args.model, args.optimizer, args.scaler, 
               args.intensity, args.channel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device).double()
    network.eval() # set in eval mode 

    with torch.no_grad():
        test_input = torch.from_numpy(data[0:]).to(device)
        test_input = Variable(test_input.to(device), requires_grad=False) 
        pred = network(test_input, device, args.future)
        # cuda tensor cannot be converted to numpy directly, 
        # tensor.cpu to copy the tensor to host memory first
        model_output = pred.detach().cpu().numpy()

    if args.scaler.lower() == "minmax":
        inp = test_input.numpy()[0,input_size:].reshape(-1,1)
        out = model_output[0,:-1].reshape(-1,1)
        plot_results(inp, out, args) # scaled
        # now inverse scaling and plots again   
        a, b = np.amin(unscaled_data[-1,:]), np.amax(unscaled_data[-1,:])
        real_inp = inp * (b - a) + a
        real_out = out * (b - a) + a
        plot_results(real_inp, real_out, args) # scaled
    elif args.scaler.lower() == "log":
        # inverse scales the log scaled validation data and model output:
        input_inverted = inv_logscale(test_input.numpy()[0,input_size:], inc)
        output_inverted = inv_logscale(model_output[0,:], inc)
        plot_results(input_inverted, output_inverted, args)


if __name__ == "__main__":
    main()
