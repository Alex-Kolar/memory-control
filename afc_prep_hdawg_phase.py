from zhinst.toolkit import Session
from zhinst.toolkit import Waveforms
import numpy as np

from afc_prep import full_waveform
from afc_prep import full_waveform_noshift


if __name__ == '__main__':
    # device params
    DEVICE_ID = 'DEV8345'
    SERVER_HOST = 'localhost'
    samp_rate = 2.4e9  # unit: Hz
    range = 5  # unit: V

    # EOM params
    Vpi = 3  # unit: V

    # waveform params
    A = Vpi / (range * np.pi)  # overall amplitude of pulse (after normalization)
    N = 1
    delta = 1e6  # unit: Hz
    tau = 2e-3  # unit: s
    beta = 10 / tau
    delta_f = 0.7e6  # unit: Hz
    f_0 = 100e6  # unit: Hz
    f_light = 195e12  # light frequency (in Hz)

    resolution = 1 / samp_rate  # unit: s
    num_points = tau / resolution
    num_points = (num_points // 16) * 16  # round to multiple of 16

    t, theta, amp = (
        full_waveform(N, delta, num_points, resolution, beta, f_light, delta_f))
    # wav = A * amp * np.sin(2 * np.pi * f_0 * t + theta)
    coeff = (1 / range) * amp
    wav = (A * theta) + coeff * np.sin(2 * np.pi * f_0 * t)
    print(f"len: {len(wav)}")

    # # TODO: delete
    # wav = np.sin(2*np.pi*100e6*t + theta + coeff * np.sin(2 * np.pi * f_0 * t))

    # connect to device
    session = Session(SERVER_HOST)
    device = session.connect_device(DEVICE_ID)

    # define waveform
    marker = np.ones_like(wav, dtype=int)
    marker[int(len(wav)/2):] = 0

    # convert to HDAWG language
    waveforms = Waveforms()
    waveforms[10] = (wav, None, marker)

    with device.set_transaction():
        device.awgs[0].write_to_waveform_memory(waveforms)
