## Load the LabOne API and other necessary packages
from zhinst.toolkit import Session
from zhinst.toolkit import Waveforms
import numpy as np

device_id = 'DEV8345'
server_host = 'localhost'

### connect to data server
session = Session(server_host)
### connect to device
device = session.connect_device(device_id)

##Generate a waveform and marker
LENGTH = 1024
wave = np.sin(np.linspace(0, 10*np.pi, LENGTH))
marker = np.concatenate([np.ones(32), np.zeros(LENGTH-32)]).astype(int)

## Upload waveforms
AWG_INDEX = 0  # use AWG 1
waveforms = Waveforms()
waveforms[10] = (wave,None,marker) # I-component wave, Q-component None, marker
device.awgs[AWG_INDEX].write_to_waveform_memory(waveforms)
