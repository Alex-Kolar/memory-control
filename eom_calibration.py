import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def eom_func(x, amplitude, omega):
    return amplitude * np.cos(omega*x) + np.abs(amplitude)


def eom_calibration(data_path):
    data = pd.read_csv(data_path)
    param_guess = [-0.8, np.pi/7.5]
    p0, cov = scipy.optimize.curve_fit(eom_func,
                                       data['volts'],
                                       data['power (mW)'],
                                       p0=param_guess)
    return p0


if __name__ == '__main__':
    DATA_PATH = 'data/eom_cal_data.csv'
    df = pd.read_csv(DATA_PATH)
    p0 = eom_calibration(DATA_PATH)

    omega = p0[1]
    v_pi = np.pi / omega
    print("V_pi:", v_pi)

    plt.figure()
    plt.scatter(df['volts'], df['power (mW)'],
                label='Data')
    v_vals = np.linspace(min(df['volts']), max(df['volts']), 100)
    plt.plot(v_vals, eom_func(v_vals, *p0),
             '--k', label='Fit')

    plt.xlabel('Voltage (V)')
    plt.ylabel('Power (mW)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
