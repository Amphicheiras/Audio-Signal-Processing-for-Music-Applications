import numpy as np
from scipy.signal import get_window
from scipy.signal.windows import blackmanharris, triang
from scipy.fft import ifft
import math
import dftModel as DFT
import utilFunctions as UF
import matplotlib.pyplot as plt
import stft as STFT

def sineModelMultiRes(x, fs, w, N, t, B):
    """
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB
    returns y: output array sound
    """
    Ns = 512  # FFT size for synthesis (even)
    H = Ns // 4  # Hop size used for analysis and synthesis
    hNs = Ns // 2  # half of synthesis FFT size
    yw = np.zeros(Ns)  # initialize output sound frame
    y = np.zeros(x.size)  # initialize output array
    sw = np.zeros(Ns)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hNs - H:hNs + H] = ow  # add triangular window
    bh = blackmanharris(Ns)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window
    for index in range(3):
        hM1 = int(math.floor((w[index].size + 1) / 2))  # half analysis window size by rounding
        hM2 = int(math.floor(w[index].size / 2))  # half analysis window size by floor
        pin = max(hNs, hM1)  # init sound pointer in middle of anal window
        pend = x.size - max(hNs, hM1)  # last sample to start a frame
        w[index] = w[index] / sum(w[index])  # normalize analysis window
        while pin < pend:  # while input sound pointer is within sound
            # -----analysis-----
            x1 = x[pin - hM1:pin + hM2]  # select frame
            mX_1, pX = DFT.dftAnal(x1, w[index], N[index])  # compute dft
            ploc = UF.peakDetection(mX_1, t)  # detect locations of peaks
            iploc, ipmag, ipphase = UF.peakInterp(mX_1, pX, ploc)  # refine peak values by interpolation
            ipfreq = fs * iploc / float(N[index])  # convert peak locations to Hertz
            # get the frequency indexes k
            k = np.where((ipfreq >= B[index][0]) & (ipfreq < B[index][1]))
            # -----synthesis-----
            Y = UF.genSpecSines(ipfreq[k], ipmag[k], ipphase[k], Ns, fs)  # generate sines in the spectrum
            fftbuffer = np.real(ifft(Y))  # compute inverse FFT
            yw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
            yw[hNs - 1:] = fftbuffer[:hNs + 1]
            y[pin - hNs:pin + hNs] += sw * yw  # overlap-add and apply a synthesis window
            pin += H  # advance sound pointer
    return y


############################################################
#####################    Sound 1    ########################
############################################################


# run the multi-res model for the first sound
fs_1, x_1 = UF.wavread("..\\..\\workspace\\A10\\A10-b-1_input.wav")
# band edges
B1_1 = (0, 1000)
B2_1 = (1000, 5000)
B3_1 = (5000, 22050)
# window sizes
M1_1 = 4095
M2_1 = 2047
M3_1 = 1023
#
N1_1 = 4096
N2_1 = 2048
N3_1 = 1024
# sinusoid threshold value
t_1 = -110
# get the windows
w1_1 = get_window('blackmanharris', M1_1)
w2_1 = get_window('blackman', M2_1)
w3_1 = get_window('hamming', M3_1)
# get the synthesized signal
y_1 = sineModelMultiRes(x_1, fs_1, [w1_1, w2_1, w3_1], [N1_1, N2_1, N3_1], t_1, [B1_1, B2_1, B3_1])
# synthesis hop size
H = 200
# synthesis FFT size
N = 4096
# y axis freqs for plotting
maxplotfreq = 22049
# STFT analysis for input sound
mX_1, _ = STFT.stftAnal(x_1, get_window('blackman', 2047), 4096, 200)

# plot the input spectrum
plt.figure(1)
plt.subplot(2,1,1)
plt.title("Original sample")
plt.xlabel("time (seconds)")
plt.ylabel("frequency (Hz)")
numFrames = int(mX_1[:,0].size)
frameTime = H*np.arange(numFrames)/float(fs_1)
binFreq = fs_1*np.arange(N*maxplotfreq/fs_1)/N
plt.pcolormesh(frameTime, binFreq, np.transpose(mX_1[:,:N*maxplotfreq//fs_1+1]))

# plot the output spectrum
mY_1, _ = STFT.stftAnal(y_1, get_window('blackman', 2047), 4096, 200)
plt.subplot(2,1,2)
plt.title("Synthesized sample")
plt.xlabel("time (seconds)")
plt.ylabel("frequency (Hz)")
numFrames = int(mY_1[:,0].size)
frameTime = H*np.arange(numFrames)/float(fs_1)
binFreq = fs_1*np.arange(N*maxplotfreq/fs_1)/N
plt.pcolormesh(frameTime, binFreq, np.transpose(mY_1[:,:N*maxplotfreq//fs_1+1]))
# plt.show()

# export the synthesized sound to a wav file
UF.wavwrite(y_1, fs_1, "..\\..\\workspace\\A10\\A10-b-1_output.wav")


############################################################
#####################    Sound 2    ########################
############################################################


# run the multi-res model for the first sound
fs_2, x_2 = UF.wavread("..\\..\\workspace\\A10\\A10-b-2_input.wav")
# band edges
B1_2 = (0, 500)
B2_2 = (500, 5000)
B3_2 = (5000, 22050)
# window sizes
M1_2 = 4095
M2_2 = 2047
M3_2 = 1023
#
N1_2 = 4096
N2_2 = 2048
N3_2 = 1024
# sinusoid threshold value
t_2 = -105
# get the windows
w1_2 = get_window('blackman', M1_2)
w2_2 = get_window('blackman', M2_2)
w3_2 = get_window('blackman', M3_2)
# get the synthesized signal
y_2 = sineModelMultiRes(x_2, fs_2, [w1_2, w2_2, w3_2], [N1_2, N2_2, N3_2], t_2, [B1_2, B2_2, B3_2])
# synthesis hop size
H = 200
# synthesis FFT size
N = 4096
# y axis freqs for plotting
maxplotfreq = 22049
# STFT analysis for input sound
mX_2, _ = STFT.stftAnal(x_2, get_window('blackman', 2047), 4096, 200)

# plot the input spectrum
plt.figure(2)
plt.subplot(2,1,1)
plt.title("Original sample")
plt.xlabel("time (seconds)")
plt.ylabel("frequency (Hz)")
numFrames = int(mX_2[:,0].size)
frameTime = H*np.arange(numFrames)/float(fs_2)
binFreq = fs_2*np.arange(N*maxplotfreq/fs_2)/N
plt.pcolormesh(frameTime, binFreq, np.transpose(mX_2[:,:N*maxplotfreq//fs_2+1]))

# plot the output spectrum
mY_2, _ = STFT.stftAnal(y_2, get_window('blackman', 2047), 4096, 200)
plt.subplot(2,1,2)
plt.title("Synthesized sample")
plt.xlabel("time (seconds)")
plt.ylabel("frequency (Hz)")
numFrames = int(mY_2[:,0].size)
frameTime = H*np.arange(numFrames)/float(fs_2)
binFreq = fs_2*np.arange(N*maxplotfreq/fs_2)/N
plt.pcolormesh(frameTime, binFreq, np.transpose(mY_2[:,:N*maxplotfreq//fs_2+1]))
plt.show()

# export the synthesized sound to a wav file
UF.wavwrite(y_2, fs_2, "..\\..\\workspace\\A10\\A10-b-2_output.wav")
