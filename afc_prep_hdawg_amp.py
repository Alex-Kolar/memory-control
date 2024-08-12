from zhinst.toolkit import Session
from zhinst.toolkit import Waveforms
import numpy as np

from afc_prep import full_waveform
from afc_prep import full_waveform_noshift


# function to convert waveform to driving voltage
def voltage(wav, v_pi):
    norm_wav = wav / np.max(np.abs(wav))
    return (v_pi / np.pi) * np.arccos(norm_wav) - (v_pi / 2)


if __name__ == '__main__':
    # device params
    DEVICE_ID = 'DEV8345'
    SERVER_HOST = 'localhost'
    samp_rate = 2.4e9  # unit: Hz
    volt_range = 5.0  # unit: V

    # EOM params
    Vpi = 7.342  # unit: V

    # waveform params
    A = 1  # overall amplitude of pulse (after normalization)
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
    # t, theta, amp = (
    #     full_waveform_noshift(N, delta, num_points, resolution, beta, 0, delta_f))
    wav_ideal = amp * np.sin(2*np.pi*f_0*t + theta)
    wav = voltage(wav_ideal, Vpi) / volt_range
    print(f"len: {len(wav)}")

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
