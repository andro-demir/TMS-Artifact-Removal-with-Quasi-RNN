import sys
import numpy as np
import scipy.io as spio

class parser:
    
    filepath = '../humanData.mat'
    start = 9990
    end = 10040

    ''' 
        Loads the .mat files for the melon data and the tms-eeg data
        MSO: Monophasic TMS-EEG data have stimulation intensities range from 
             10 to 80 (increments of 10)
        channel: There are 63 channels, tagged as (0,1,..62)
        start: index of the first sample. Samples before start are truncated
        end: index of the first sample. Samples after start are truncated
    '''
    def __init__(self):
        try:
            self.eeg_data = spio.loadmat(parser.filepath, squeeze_me=True)
            print("TMS-EEG data is successfully loaded.")
        except Exception:
            print("Sorry, %s does not exist." %parser.filepath)
            print("Check the filepath.")
            sys.exit(1)  

    def get_intensity(self, intensity):
        self.MSO = 'MSO%d'%intensity
        self.intensity_data = self.eeg_data[self.MSO]
    
    def get_channel(self, channel):
        self.channel = channel
        self.channel_data = self.intensity_data[self.channel, 
                                                parser.start:parser.end, :]
               