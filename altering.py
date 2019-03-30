from __future__ import division
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import datetime
import numpy
import time
import math
plt.style.use('ggplot')
global tMax, tTCR, tTRN, tIN, vThr, sigma, rRetToTCR, rRetToIN, rInToTCR, r_TRN_to_TCR, T, timeList, iRetToTCR, iRetToIN, iInToTCR, iInToIn, i_TRN_to_TCR, i_TRN_to_TRN, betaAmpa, betaGaba, vTCR, vIN, iLeakTCR, iLeakIN, gAmpaRetToTCR, gAmpaRetToIN, gGabaInToTCR, EAmpa, EGaba, EGaba_TRN_or_IN_to_TCR, EGaba_TRN_to_TRN_or_IN_to_IN, cIRE, cTRE, cISI, cTII, cNSI, gLeakIN, gLeakTCR, ELeakIN, ELeakTCR, r_TCR_to_TRN, iLeakTRN, cTNI, gLeakTRN, ELeakTRN
tTCR = [0] * 10000
tTRN = [0] * 10000
tIN = [0] * 10000
rRetToTCR = [0] * 10000
r_TCR_to_TRN = [0] * 10000
r_TRN_to_TCR = [0] * 10000
rInToTCR = [0] * 10000
rRetToIN = [0] * 10000
iRetToTCR = [0] * 10000
iRetToIN = [0] * 10000
i_TCR_to_TRN = [0] * 10000
iInToTCR = [0] * 10000
i_TRN_to_TCR = [0] * 10000
i_TRN_to_TRN = [0] * 10000
iInToIn = [0] * 10000
timeList = np.arange(0, 10.000, 0.001)
betaAmpa = 50
betaGaba = 40
vTCR = [0] * 10000
vIN = [0] * 10000
vTRN = [0] * 10000
iLeakTCR = [0] * 10000
iLeakIN = [0] * 10000
iLeakTRN = [0] * 10000
iLeakTCR[0] = 0.0001
iLeakIN[0] = 0.0001
iLeakTRN[0] = 0.0001
EAmpa = 0
EGaba_TRN_or_IN_to_TCR = -85
EGaba_TRN_to_TRN_or_IN_to_IN = -75
cIRE = 47.4
cTRE = 7.1
cTNI = 11.5875
cISI = 23.6
cNTE = 35
cTII = 15.45
cNSI = 20
gLeakTCR = 10
gLeakIN = 10
gLeakTRN = 10
ELeakTCR = -55
ELeakIN = -72.5
ELeakTRN = -72.5
vTCR[0] = -65
vIN[0] = -75
vTRN[0] = -65
# foll will be altered
gAmpaRetToTCR = 300
gAmpaRetToIN = 100
gAmpaTCRToTRN = 100
gGaba = 100
gGabaInToTCR = 100


tMax = 1
vThr = -32
sigma = 3.7
rRetToTCR[0] = (0.002)
rRetToIN[0] = 0.001
r_TCR_to_TRN[0] = (0.003)
rInToTCR[0] = 0.004
r_TRN_to_TCR[0] = 0.005
rRetToIN[0] = (0.002)


def vRetinaInitialisation():
    global a, b
    mean = -65
    standardDeviation = 2
    variance = math.pow(standardDeviation, 2)
    a = (mean - (variance / 2))
    b = ((mean * 2) - a)


def generateRandomUniformNumbers():
    global vRet
    vRet = np.random.uniform(a, b, 10000)


def plotNoise():
    plt.plot(timeList, vRet)
    plt.show()


def computeTRet():  # equation 5 tRet
    global tRet
    tRet = []

    for i in range(0, len(vRet)):
        tRet.append(tMax / (1 + math.exp(-((vRet[i] - vThr) / sigma))))


def fAmpa(X, Y, ij):  # equation 6 using runge kutta, ampa connection
    #alpha = 1000
    # return (3 * math.exp((-1*X)) - (0.4 * Y))
    if(np.isnan(Y)):
        print("the ith value of rRetToTCR is causing trouble", ij)
        exit()
    return ((1000 * X * (1 - Y)) - (betaAmpa * Y))


def fGaba(X, Y, ij):  # equation 6 using runge kutta, gaba connection
    #alpha = 1000
    # return (3 * math.exp((-1*X)) - (0.4 * Y))
    if(np.isnan(Y)):
        print("the ith value of rRetToTCR is causing trouble", ij)
        exit()
    return ((1000 * X * (1 - Y)) - (betaGaba * Y))


def functionRungeKutta():
    # get done with steps, there is no need of steps.
    h = 0.001
    tTCR[0] = ((tMax) / (1 + math.exp(-1 * ((vTCR[0] - vThr) / sigma))))
    tIN[0] = (tMax / (1 + math.exp(-((vIN[0] - vThr) / sigma))))
    tTRN[0] = ((tMax) / (1 + math.exp(-1 * ((vTRN[0] - vThr) / sigma))))
    iRetToTCR[0] = (gAmpaRetToTCR * rRetToTCR[0] * (vTCR[0] - EAmpa) * cTRE)
    iRetToIN[0] = (gAmpaRetToIN * rRetToIN[0] * (vIN[0] - EAmpa) * cIRE)
    i_TCR_to_TRN[0] = (gAmpaTCRToTRN * r_TCR_to_TRN[0]
                       * (vTRN[0] - EAmpa) * cNTE)
    iInToTCR[0] = (gGaba * rInToTCR[0] *
                   (vTCR[0] - EGaba_TRN_or_IN_to_TCR) * cTII)
    iInToIn[0] = (gGaba * rInToTCR[0] * (vIN[0] -
                                         EGaba_TRN_to_TRN_or_IN_to_IN) * cISI)
    i_TRN_to_TCR[0] = (gGabaInToTCR * r_TRN_to_TCR[0] *
                       (vTCR[0] - EGaba_TRN_or_IN_to_TCR) * cTNI)
    i_TRN_to_TRN[0] = (gGaba * r_TRN_to_TCR[0] *
                       (vTRN[0] - EGaba_TRN_to_TRN_or_IN_to_IN) * cNSI)

    for i in range(0, 9999):
        # all the rS
        rRetToTCR[i+1] = rRetToTCR[i] + h * fAmpa(tRet[i], rRetToTCR[i], i)
        rRetToIN[i+1] = rRetToIN[i] + h * fAmpa(tRet[i], rRetToIN[i], i)
        r_TCR_to_TRN[i+1] = r_TCR_to_TRN[i] + \
            h * fAmpa(tTCR[i], r_TCR_to_TRN[i], i)
        rInToTCR[i+1] = rInToTCR[i] + h * fGaba(tIN[i], rInToTCR[i], i)
        r_TRN_to_TCR[i+1] = r_TRN_to_TCR[i] + \
            h * fGaba(tTRN[i], r_TRN_to_TCR[i], i)
        # all the iS
        iRetToTCR[i+1] = gAmpaRetToTCR * \
            rRetToTCR[i+1] * (vTCR[i] - EAmpa) * cTRE
        iRetToIN[i+1] = gAmpaRetToIN * rRetToIN[i+1] * (vIN[i] - EAmpa) * cIRE
        iInToTCR[i+1] = gGaba * rInToTCR[i+1] * \
            (vTCR[i] - EGaba_TRN_or_IN_to_TCR) * cTII
        iInToIn[i+1] = gGaba * rInToTCR[i+1] * \
            (vIN[i] - EGaba_TRN_to_TRN_or_IN_to_IN) * cISI
        i_TCR_to_TRN[i+1] = gAmpaTCRToTRN * \
            r_TCR_to_TRN[i+1] * (vTRN[i] - EAmpa) * cNTE
        # gGabaInToTCR is the same for this connection, hence, used here.
        i_TRN_to_TCR[i+1] = gGabaInToTCR * r_TRN_to_TCR[i+1] * \
            (vTCR[i] - EGaba_TRN_or_IN_to_TCR) * cTNI
        i_TRN_to_TRN[i+1] = (gGaba * r_TRN_to_TCR[i+1] *
                             (vTRN[i] - EGaba_TRN_to_TRN_or_IN_to_IN) * cNSI)
        # all the iLeaks
        iLeakTCR[i+1] = ((gLeakTCR * (vTCR[i] - (ELeakTCR))))
        iLeakIN[i+1] = ((gLeakIN * (vIN[i] - (ELeakIN))))
        iLeakTRN[i+1] = ((gLeakTRN * (vTRN[i] - ELeakIN)))
        # all the v
        vTCR[i+1] = vTCR[i] + h * \
            (-1 * (iRetToTCR[i] + iInToTCR[i] + i_TRN_to_TCR[i] + iLeakTCR[i]))
        vIN[i+1] = vIN[i] + h * (-1 * (iRetToIN[i] + iInToIn[i] + iLeakIN[i]))
        # adding trn to tcr's current because that is the same value for iTRN_to_TRN
        vTRN[i+1] = vTRN[i] + h * \
            (-1 * (i_TCR_to_TRN[i] + i_TRN_to_TRN[i] + iLeakTRN[i]))
        # the remaining tS
        tTCR[i+1] = ((tMax) / (1 + math.exp(-1 * ((vTCR[i+1] - vThr) / sigma))))
        tIN[i+1] = ((tMax) / (1 + math.exp(-1 * ((vIN[i+1] - vThr) / sigma))))
        tTRN[i+1] = ((tMax) / (1 + math.exp(-1 * ((vTRN[i] - vThr) / sigma))))


def alter_gmax():

    max_freq = 3  # i assume that this is the target freq i need to achieve.
    # it gives freq and power corresponding to it
    f, Pxx_spec = signal.welch(vTCR, 1000, 'hanning', 500, scaling='spectrum')
    # welch converts the timeseries data into frequency based data using fast fourier transform; it changes the point difference values while processing and hence the initial freq is ranging from 5 to 8.
    calulated_max_freq = f[list(Pxx_spec).index(max(Pxx_spec))]

    # if true, add values otherwise no
    gAmpaRetToTCR_change = True
    gAmpaRetToIN_change = True
    gAmpaTCRToTRN_change = True
    gGaba_change = True
    gGabaInToTCR_change = True

    global gAmpaRetToTCR, gAmpaTCRToTRN, gAmpaRetToIN, gGaba, gGabaInToTCR

    original_gAmpaRetToTCR = gAmpaRetToTCR
    original_gAmpaTCRToTRN = gAmpaTCRToTRN
    original_gAmpaRetToIN = gAmpaRetToIN
    original_gGaba = gGaba
    original_gGabaInToTCR = gGabaInToTCR

    present_diff = abs(max_freq - calulated_max_freq)
    # our objective is to lessen the absolute difference.
    print(
        'Need to Achive: {}, From: {}'.format(
            max_freq, calulated_max_freq
        )
    )
    # can change it to 0.5 for more accurancy, as of now, 1 diff is the threshold.
    while present_diff > 1:

        if gAmpaRetToTCR_change:
            gAmpaRetToTCR += 1
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            f, Pxx_spec = signal.welch(
                vTCR, 1000, 'hanning', 500, scaling='spectrum')
            calulated_max_freq = f[list(Pxx_spec).index(max(Pxx_spec))]
            if abs(calulated_max_freq - max_freq) > present_diff:
                gAmpaRetToTCR -= 2
                gAmpaRetToTCR_change = False
        else:
            gAmpaRetToTCR -= 1

        if gAmpaRetToIN_change:
            gAmpaRetToIN += 1
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            f, Pxx_spec = signal.welch(
                vTCR, 1000, 'hanning', 500, scaling='spectrum')
            calulated_max_freq = f[list(Pxx_spec).index(max(Pxx_spec))]
            if abs(calulated_max_freq - max_freq) > present_diff:
                gAmpaRetToIN -= 2
                gAmpaRetToIN_change = False
        else:
            gAmpaRetToTCR -= 1

        if gAmpaTCRToTRN_change:
            gAmpaTCRToTRN += 0.1
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            f, Pxx_spec = signal.welch(
                vTCR, 1000, 'hanning', 500, scaling='spectrum')
            calulated_max_freq = f[list(Pxx_spec).index(max(Pxx_spec))]
            if abs(calulated_max_freq - max_freq) > present_diff:
                gAmpaTCRToTRN -= 0.2
                gAmpaTCRToTRN_change = False
        else:
            gAmpaTCRToTRN -= 0.1

        if gGaba_change:
            gGaba += 0.1
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            f, Pxx_spec = signal.welch(
                vTCR, 1000, 'hanning', 500, scaling='spectrum')
            calulated_max_freq = f[list(Pxx_spec).index(max(Pxx_spec))]
            if abs(calulated_max_freq - max_freq) > present_diff:
                gGaba -= 0.2
                gGaba_change = False
        else:
            gGaba -= 0.1

        if gGabaInToTCR_change:
            gGabaInToTCR += 0.1
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            vRetinaInitialisation()
            generateRandomUniformNumbers()
            computeTRet()
            functionRungeKutta()
            f, Pxx_spec = signal.welch(
                vTCR, 1000, 'hanning', 500, scaling='spectrum')
            calulated_max_freq = f[list(Pxx_spec).index(max(Pxx_spec))]
            if abs(calulated_max_freq - max_freq) > present_diff:
                gGabaInToTCR -= 0.2
                gGabaInToTCR_change = False
        else:
            gGabaInToTCR -= 0.1

        vRetinaInitialisation()
        generateRandomUniformNumbers()
        computeTRet()
        functionRungeKutta()
        vRetinaInitialisation()
        generateRandomUniformNumbers()
        computeTRet()
        functionRungeKutta()
        f, Pxx_spec = signal.welch(
            vTCR, 1000, 'hanning', 500, scaling='spectrum')
        calulated_max_freq = f[list(Pxx_spec).index(max(Pxx_spec))]
        new_dis = abs(calulated_max_freq - max_freq)
        print(
            'Old Distance: {} New Distance: {}'.format(
                present_diff, new_dis
            )
        )
        present_diff = new_dis

    print('Original g_max terms: ')
    print (
        "original_gAmpaTCRToTRN: {}, original_gAmpaRetToIN: {}, original_gAmpaRetToTCR: {},original_gGaba: {}, original_gGabaInToTCR: {}".format(
            original_gAmpaTCRToTRN, original_gAmpaRetToIN, original_gAmpaRetToTCR, original_gGaba, original_gGabaInToTCR
        )
    )
    print('Altered g_max terms: ')
    print(
        "gAmpaTCRToTRN: {}, gAmpaRetToIN: {}, gAmpaRetToTCR: {}, gGaba: {}, gGabaInToTCR: {}".format(
            gAmpaTCRToTRN, gAmpaRetToIN, gAmpaRetToTCR, gGaba, gGabaInToTCR
        )
    )
    print 
    f, Pxx_spec = signal.welch(vTCR, 1000, 'hanning', 500, scaling='spectrum')
    plt.figure()
    plt.semilogy(f, Pxx_spec, linewidth=2.0)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.title('Power spectrum (scipy.signal.welch)')
    plt.show()


if __name__ == "__main__":
    vRetinaInitialisation()
    generateRandomUniformNumbers()
    computeTRet()
    functionRungeKutta()
    alter_gmax()
