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

from afc_prep import full_waveform


if __name__ == '__main__':
    # device params
    DEVICE_ID = 'DEV8345'
    SERVER_HOST = 'localhost'
    samp_rate = 2.4e9  # (Hz)
    volt_range = 5.0  # (V)

    # EOM params
    Vpi = 3.9  # (V)

    # waveform params
    A = Vpi / (volt_range * np.pi)  # overall amplitude of pulse (after normalization)
    N = 99
    delta = 1e6  # (Hz)
    tau = 2e-3  # (s)
    beta = 10 / tau  # (Hz)
    delta_f = 0.7e6  # (Hz)
    f_0 = 100e6  # center frequency (Hz)
    f_light = 195e12  # light frequency (Hz)

    # calculation of waveform
    resolution = 1 / samp_rate  # (s)
    num_points = tau / resolution
    num_points = (num_points // 16) * 16  # round to multiple of 16

    t, theta, amp = (
        full_waveform(N, delta, num_points, resolution, beta, f_light, delta_f))
    coeff = amp / (volt_range * np.max(amp))  # maximum modulation when max voltage is about 1
    wav = coeff * np.sin(2*np.pi*f_0*t + theta)  # V(t)
    print(f"len: {len(wav)}")

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
