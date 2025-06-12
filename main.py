from zhinst.toolkit import Session
from zhinst.toolkit import Waveforms
import numpy as np

from constants import *
from afc_prep import full_waveform
from afc_prep_hdawg_amp import amp_to_volts


if __name__ == '__main__':
    """
    Define parameters
    """

    # overall script params

    # burning waveform params
    N = 1
    delta = 1e6  # unit: Hz
    tau = 2e-3  # unit: s
    beta = 10 / tau
    delta_f = 0.7e6  # unit: Hz
    f_0 = 100e6  # unit: Hz
    f_light = 195e12  # light frequency (in Hz)

    # device params
    samp_rate = 2.4e9  # unit: Hz
    volt_range = 5.0  # unit: V

    # EOM params
    Vpi = 7.342  # unit: V

    """
    Waveform Generation
    """
    # waveform for burning sequence
    resolution = 1 / samp_rate  # unit: s
    num_points = tau / resolution
    num_points = (num_points // 16) * 16  # round to multiple of 16

    t, theta, amp = (
        full_waveform(N, delta, num_points, resolution, beta, f_light, delta_f))
    wav_ideal = amp * np.sin(2 * np.pi * f_0 * t + theta)
    wav_burn = amp_to_volts(wav_ideal, Vpi) / volt_range
    print(f'len: {len(wav_burn)}')

    # add waveforms
    wav = wav_burn

    """
    Upload Waveform
    """
    # connect to device
    session = Session(SERVER_HOST)
    device = session.connect_device(DEVICE_ID)

    # define waveform
    marker = np.ones_like(wav, dtype=int)
    marker[int(len(wav) / 2):] = 0

    # convert to HDAWG language
    waveforms = Waveforms()
    waveforms[10] = (wav, None, marker)

    with device.set_transaction():
        device.awgs[0].write_to_waveform_memory(waveforms)
