from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import log
import torch
from rnn_model import Temporal_Learning, set_optimization, train_model, \
                      validate_model, test_model, save_model
from melon_data_parser import parser
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter


'''
    Trains network using GPU, if available. Otherwise uses CPU.
'''
def set_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: %s\n" %device)
    # .double() will make sure that  MLP will process tensor
    # of type torch.DoubleTensor:
    return model.to(device).double(), device

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-save", type=str2bool, help=("Save model after " 
                        "training ('True' or 'False')"), required=True)
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
    assert args.save == True or args.save == False, ("\nAcceptable entries for"
           " argument save are True, False, y, n, t, f, 1, 0. You entered: " +
           args.Save)
    assert args.model.lower() == "lstm" or args.model.lower() == "gru", ("\n"
           "Acceptable entries for argument model are: 'lstm' and 'gru'\nYou"
           " entered: " + args.model)
    assert args.optimizer.lower() == 'l-bfgs' or \
           args.optimizer.lower() == 'adam', ("\nAcceptable entries for " 
           "optimizer are l-bfgs and adam. You entered: " + args.optimizer)
    assert args.future > 0, "Future must be a positive integer."
    assert args.intensity in acceptable_MSO, ("Acceptable entries for TMS "
           "intensity (MSO) are 0, 10, 20, 30, 40, 50, 60, 70, 80.\nYou "
           "entered "+ args.intensity)
    assert args.channel in acceptable_channel, ("Acceptable entries for the "
           "EEG channels are 0, 1, 2, 3, ... 62.\nYou entered " + args.channel)
    assert args.scaler in acceptable_scalers, ("Acceptable entries for the "
           "scaling method are 'minmax' and 'log'.\nYou entered " + args.scaler)
    return args

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

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))

"""
    Splits the trials into train, test and validation sets.
    Inputs take the the entire array
    Outputs begin from the index input_size
    So the model can always look back input_size number of samples
    for training the rnn
""" 
def create_dataset(data, input_size, device, k, fold):
    indices = list(range(data.shape[0]))
    fold_indx = list(split(indices, k))
    train_indx = list(set(indices) - set(fold_indx[fold]))
    train_data = data[np.ix_(train_indx)]

    train_input = torch.from_numpy(train_data).to(device)
    train_output = torch.from_numpy(train_data[:,input_size:]).to(device)
    
    val_input = torch.from_numpy(data[fold_indx[fold][:-1]]).to(device)
    val_output = torch.from_numpy(data[fold_indx[fold][:-1], input_size:]).\
                                                                to(device)

    test_input = torch.from_numpy(data[fold_indx[fold][-1]:]).to(device)
    test_output = torch.from_numpy(data[fold_indx[fold][-1]:, input_size:]).\
                                                                  to(device)
    
    return train_input, train_output, val_input, val_output,\
           test_input, test_output
 
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
    plt.savefig('MSO%s_ch%s_%s_%s.pdf'%(args.intensity, args.channel, 
                args.model.lower(), args.optimizer.lower()))
    plt.show()

def main():
    args = pass_legal_args()
    dropout = 0.5
    hidden_size, input_size = 64, 5
       
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

    # This implements K-fold cross validation, K=10
    k = 5
    for fold in range(k): 
        writer = SummaryWriter() # to plot the loos graph on tensorboard

        # Builds the model, sets the device
        temporal_model = Temporal_Learning(args.model, input_size, hidden_size,
                                           dropout)
        temporal_model, device = set_device(temporal_model)

        # Splits the data for train/validation/test input/output
        train_input, train_output, val_input, val_output, \
        test_input, test_output = create_dataset(data, input_size, device, k, 
                                                                        fold)
        criterion, optimizer, epochs = set_optimization(temporal_model, 
                                                        args.optimizer)  
    
        for epoch in range(epochs):
            print('Epoch: ', epoch+1)
            train_model(temporal_model, train_input, train_output, optimizer, 
                        epoch, criterion, device, writer)
            validate_model(temporal_model, val_input, val_output, epoch, 
                           criterion, args.future, device, writer)           
            model_output = test_model(temporal_model, test_input, test_output, 
                                      epoch, criterion, args.future, device, 
                                      writer) 

        if args.save == True:
            save_model(temporal_model, args.optimizer.lower(), 
                       args.model.lower(), args.scaler, args.intensity, 
                       args.channel)

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

        writer.close()

if __name__ == "__main__":
    main()