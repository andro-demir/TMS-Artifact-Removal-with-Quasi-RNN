'''
    This code was forked from: 
    https://github.com/mila-udem/summerschool2016/blob/master/theano/ 
                                        rnn_tutorial/synthetic.py
    Mackey Glass equation is a nonlinear time delay differential equation.
    It is used in abstractions that implement noisy and chaotic time series
    predictions.
    Multiple Sinewave Oscillator generates time-series, a sum of two sines
    with incommensurable periods.
    Lorentz Attractor is created by Runge-Kutta integration of the Lorenz
    equations. "Chaotic"
'''

import collections
import numpy as np
import matplotlib.pyplot as plt

'''
    Generates n_samples arbitrary sine waves of length sample_len samples.
    Dummy data.
'''
def sinusoid(sample_len=1000, T=20, n_samples=100):
    x = np.empty((n_samples, sample_len), 'int64')
    x[:] = np.array(range(sample_len)) + \
           np.random.randint(-4 * T, 4 * T, n_samples).reshape(n_samples, 1)
    y = np.sin(x / 1.0 / T).astype('float64')
    return y

'''
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generates the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
'''
def mackey_glass(sample_len=1000, tau=17, seed=None, n_samples=100):
    delta_t = 10
    history_len = tau * delta_t
    # Initial conditions for the history of the system
    timeseries = 1.2
    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len,1))

        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                             0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries

        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)
    
    return np.asarray(samples).reshape((n_samples, sample_len))

'''
    mso(sample_len=1000, n_samples = 1) -> input
    Generate the Multiple Sinewave Oscillator time-series, a sum of two sines
    with incommensurable periods. Parameters are:
        - sample_len: length of the time-series in timesteps
        - n_samples: number of samples to generate
'''
def mso(sample_len=2000, n_samples=100):
    signals = []
    for _ in range(n_samples):
        phase = np.random.rand()
        x = np.atleast_2d(np.arange(sample_len)).T
        signals.append(np.sin(0.2 * x + phase) + np.sin(0.311 * x + phase))
    
    return np.asarray(signals).reshape((n_samples, sample_len))

"""
    This function generates a Lorentz time series of length sample_len,
    with standard parameters sigma, rho and beta.
    Couldn't be reproduced to 100 samples, hence not used in the lstm test.
"""
def lorentz(sample_len=1000, sigma=10, rho=28, beta=8/3, step=0.01):
    x = np.zeros([sample_len])
    y = np.zeros([sample_len])
    z = np.zeros([sample_len])
    # Initial conditions taken from 'Chaos and Time Series Analysis', J. Sprott
    x[0] = 0
    y[0] = -0.01
    z[0] = 9
    for t in range(sample_len - 1):
        x[t + 1] = x[t] + sigma * (y[t] - x[t]) * step
        y[t + 1] = y[t] + (x[t] * (rho - z[t]) - y[t]) * step
        z[t + 1] = z[t] + (x[t] * y[t] - beta * z[t]) * step

    x.shape += (1,)
    y.shape += (1,)
    z.shape += (1,)
    return np.concatenate((x, y, z), axis=1)


# Unit tests the noisy and chaotic time series:
if __name__ == "__main__":
    mg_samples = mackey_glass()
    print(mg_samples.shape)
    ms_samples = mso()
    print(ms_samples.shape)
    mg_samples = mackey_glass()[0]
    mso_samples = mso()[0]
    l_samples = lorentz()
    print(l_samples.shape)
    f, axarr = plt.subplots(5, sharex=True)
    axarr[0].set_title('Mackey Glass Time Series')
    axarr[0].plot(mg_samples)
    axarr[1].set_title('Multiple Sine Wave Oscillator')
    axarr[1].plot(mso_samples)
    axarr[2].set_title('Lorentz Time Series')
    axarr[2].plot(l_samples[:, 0])
    axarr[3].set_title('Lorentz Time Series')
    axarr[3].plot(l_samples[:, 1])
    axarr[4].set_title('Lorentz Time Series')
    axarr[4].plot(l_samples[:, 2])
    plt.show()

    print("Chaotic Time Series Run Successful!")