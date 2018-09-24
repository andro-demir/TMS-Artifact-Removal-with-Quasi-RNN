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
    Helper function to set the loss function, optimization and learning rate.
    Factor: by which the learning rate will be reduced
'''
def set_optimization(model, optimizer):
    criterion = nn.MSELoss()
    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        epochs = 200
    # L-BFGS is also well suited if we can load all data to train and the 
    # optimization requires to solve large number of variables
    elif optimizer.lower() == 'l-bfgs':
        optimizer = optim.LBFGS(model.parameters(), lr=0.5)
        epochs = 80
    return criterion, optimizer, epochs
    
'''
    Sets the model in training mode. 
'''    
def train_model(model, input, output, optimizer, epoch, criterion, device,
                writer):
    input = Variable(input.to(device), requires_grad=True) 
    output = Variable(output.to(device), requires_grad=False)
    # Sets the model in training mode. This corrects for the differences
    # in dropout, batch normalization during training and testing:
    model.train()
    #writer.add_graph(model, (input))
    # Gradients of all params = 0:
    model.zero_grad() # resets grads
    def closure():
        optimizer.zero_grad()
        pred = model(input, device)
        loss = criterion(pred, output)
        print('Training Loss:', loss.item())
        writer.add_scalar('Training Loss', loss.item(), epoch)
        loss.backward()
        return loss
    
    optimizer.step(closure)

'''
    Sets the model in validation mode. Tracks the loss graph.
    Compares the loss with training to avoid overfitting.
'''
def validate_model(model, val_input, val_output, epoch, criterion, 
                   future, device, writer):
    # This corrects for the differences in dropout, batch normalization
    # during training and testing:
    model.eval()
    # torch.no_grad() disables gradient calculation 
    # It is useful for inference. Since we are not doing backprop in testing,
    # it reduces memory consumption for computations that would otherwise 
    # have requires_grad=True. You can also add volatile=True inside 
    # Variable(test_X.to(device)) as an additional parameter:
    with torch.no_grad():
        val_input = Variable(val_input.to(device), requires_grad=False) 
        val_output = Variable(val_output.to(device), requires_grad=False)
        pred = model(val_input, device, future)
        loss = criterion(pred[:, :-future], val_output)        
        print('Validation Loss:', loss.item())
        writer.add_scalar('Validation Loss', loss.item(), epoch)

'''
    Sets the model in testing mode. 
'''
def test_model(model, test_input, test_output, epoch, criterion, future, 
               device, writer):
    model.eval()
    with torch.no_grad():
        test_input = Variable(test_input.to(device), requires_grad=False) 
        test_output = Variable(test_output.to(device), requires_grad=False)
        pred = model(test_input, device, future)
        loss = criterion(pred[:, :-future], test_output)        
        print('Test Loss:', loss.item())
        writer.add_scalar('Test Loss', loss.item(), epoch)
        print(50 * '-')
        # cuda tensor cannot be converted to numpy directly, 
        # tensor.cpu to copy the tensor to host memory first
        model_output = pred.detach().cpu().numpy()
    
    return model_output

'''
    Helper function to save the trained model.
    Called in main()
'''
def save_model(model, optimizer, rnn_type, scaler, intensity, channel):
    torch.save(model.state_dict(), f="../TrainedModels/tmseeg_" + rnn_type +
                                     "_" + optimizer + "_" + scaler + "_" + 
                                     intensity + "_" + channel + "_.model")
    
    