"""Code to generate an AFC using the HDAWG and a phase EOM.

BEFORE CODE CAN RUN SUCCESSFULLY:
Upload and save the following code to the HDAWG

const LENGTH = 4800000;                    // Replace with the length output from file
wave w = placeholder(LENGTH, true, false); // Create a waveform of size LENGTH, with one marker
assignWaveIndex(1, w, 10);                 // Create a wave table entry with placeholder waveform
                                           // routed to output 1, with index 10
playWave(1, w);
"""

from zhinst.toolkit import Session
from zhinst.toolkit import Waveforms
import numpy as np
import os

from constants import DEVICE_ID, SERVER_HOST


if __name__ == '__main__':
    # saving params
    FILENAME = "/Users/alexkolar/Desktop/Lab/memory-control/waveform_data/waveform_5_500MHz.bin"
    if not os.path.exists(FILENAME):
        raise ValueError(f"File '{FILENAME}' does not exist")

    with open(FILENAME) as file:
        wav = np.fromfile(file)

    # connect to device
    session = Session(SERVER_HOST)
    device = session.connect_device(DEVICE_ID)

    # define waveform marker
    marker = np.ones_like(wav, dtype=int)
    marker[int(len(wav)/2):] = 0

    # convert to HDAWG language
    waveforms = Waveforms()
    waveforms[10] = (wav, None, marker)

    with device.set_transaction():
        device.awgs[0].write_to_waveform_memory(waveforms)
