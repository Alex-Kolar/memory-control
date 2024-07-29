from zhinst.toolkit import Session
from zhinst.toolkit import Waveforms
import numpy as np

from afc_prep import full_waveform
from eom_calibration import eom_calibration


def power_to_volts(power, p0):
    amplitude, omega = p0
    power = np.abs(amplitude) * power
    return np.arccos((power - np.abs(amplitude)) / amplitude) / omega


if __name__ == '__main__':
    CONVERT_POWER = True
    DATA_DIR = 'data/eom_cal_data.csv'

    DEVICE_ID = 'DEV8345'
    SERVER_HOST = 'localhost'

    samp_rate = 2.4e9  # unit: Hz

    A = 0.5  # overall amplitude of pulse (after normalization)
    N = 5
    delta = 1e6  # unit: Hz
    tau = 2e-3  # unit: s
    beta = 10 / tau
    delta_f = 0.7e6

    resolution = 1 / samp_rate  # unit: s
    num_points = tau / resolution
    num_points = (num_points // 16) * 16

    f_0 = 100e6  # unit: Hz
    f_light = 195e12  # light frequency (in Hz)

    t, theta, amp, c1, c2 = (
        full_waveform(N, delta, num_points, resolution, beta, f_0, delta_f))
    wav = amp * np.sin(2 * np.pi * f_0 * t + theta)
    wav = wav / np.abs(np.max(wav))  # normalize
    print(f"len: {len(wav)}")

    # if necessary, convert to power and feed through modulator calibration
    if CONVERT_POWER:
        calibration_data = eom_calibration(DATA_DIR)
        out = power_to_volts(wav ** 2, calibration_data)
    else:
        out = wav

    # connect to device
    session = Session(SERVER_HOST)
    device = session.connect_device(DEVICE_ID)

    # define waveform
    marker = np.ones_like(out, dtype=int)
    marker[int(len(out) / 2):] = 0

    # convert to HDAWG language
    waveforms = Waveforms()
    waveforms[10] = (out, None, marker)
    print(waveforms)
    with device.set_transaction():
        device.awgs[0].write_to_waveform_memory(waveforms)
        print("upload complete")
