import scipy.io as spio
import sys

# TO DO

def load_dataset(): 
    try:
        # Monophasic TMS-EEG date with stimulation intensities ranging from 
        # 10 to 80:
        for i in range(10, 80):
            eeg_data%d %(i) = spio.loadmat('tmseegData.mat', squeeze_me=True)[i]
        print("TMS-EEG data is successfully loaded.")
    except Exception:
        print("Sorry, tmseegData.mat does not exist.")
        sys.exit(1)   
    finally:
        print('-' * 15)
    return eeg_data
