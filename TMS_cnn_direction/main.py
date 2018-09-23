import time
import torch
import torch.nn as nn
import torch.optim as optim
import ConvNet

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# TO DO:
# Load Partition and Manipulate the dataset 

def showTime(elapsedTime):
	minutes, seconds = divmod(elapsedTime, 60)
	hours, minutes = divmod(minutes, 60)
	print("Elapsed time is: %d hours %d minutes %d seconds." 
    	  %(hours, minutes, seconds))

def main():
    startTime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet.Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 90

    for epoch in range(1, epochs + 1):
        ConvNet.train(model, device, train_data, criterion, optimizer, epoch)
        ConvNet.test(model, device, criterion, test_data)
    
    print(50 * "-")
    print('Finished Training')
    endTime = time.time()
    elapsedTime = int(endTime-startTime)
    showTime(elapsedTime)


if __name__ == '__main__':
    main()

