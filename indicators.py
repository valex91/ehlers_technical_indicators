from math import pi
import numpy as np
import pandas_ta as ta


def EhlersEvenBetterSineWeave(df, hpLength=40, SSFLength=10):
    dataframe = df.copy()
    dataframe['alpha1'] = (1 - np.sin(pi / hpLength)) / np.cos(2 * pi / hpLength)
    dataframe['alpha2'] = np.exp(-1.414 * pi / SSFLength)
    dataframe['beta'] = 2 * dataframe['alpha2'] * np.cos(1.414 * pi / SSFLength)
    dataframe['c3'] = -dataframe['alpha2'] * dataframe['alpha2']
    dataframe['c1'] = 1 - dataframe['beta'] - dataframe['c3']

    dataframe['hp'] = .0
    dataframe['hp'] = (0.5 * (1 + dataframe['alpha1']) * (dataframe['close'] - dataframe['close'].shift(1))) + (
            dataframe['alpha1'] * dataframe['hp'].shift(1))

    dataframe['filter'] = .0
    dataframe['filter'] = (dataframe['c1'] * (dataframe['hp'] + dataframe['hp'].shift(1)) / 2) + (
            dataframe['beta'] * dataframe['filter'].shift(1)) + (dataframe['c3'] * dataframe['filter'].shift(2))
    dataframe['wave'] = (dataframe['filter'] + dataframe['filter'].shift(1) + dataframe['filter'].shift(2)) / 3
    dataframe['pwr'] = ((dataframe['filter'] * dataframe['filter']) + (
            dataframe['filter'].shift(1) * dataframe['filter'].shift(1)) + (
                                dataframe['filter'].shift(2) * dataframe['filter'].shift(2))) / 3
    dataframe['wave'] = dataframe['wave'] / np.sqrt(dataframe['pwr'])

    dataframe['signal'] = np.where(dataframe['wave'].gt(0), 1, np.where(dataframe['wave'].lt(0), -1, 0))

    return dataframe['wave'], dataframe['signal']


def EhlersCCIInverseFisherTransform(df, Length=20, smoothingLength=9):
    dataframe = df.copy()

    dataframe['CCI'] = df.ta.cci(close=dataframe['close'], length=Length)
    dataframe['firstWeight'] = .1 * (dataframe['CCI'])
    dataframe['secondWeight'] = dataframe.ta.wma(close=dataframe['firstWeight'], length=smoothingLength)
    dataframe['inverseFisher'] = (
            (np.expm1(2 * dataframe['secondWeight'])) / (np.exp(2 * dataframe['secondWeight']) + 1))

    return dataframe['inverseFisher']


def ElherIstantaneousTrendline(df, alpha=0.7):
    dataframe = df.copy()
    dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2

    dataframe['itt'] = (dataframe['hl2'] + 2 * dataframe['hl2'].shift(1) + dataframe['hl2'].shift(2)) / 4

    dataframe['itt'] = (alpha - alpha * alpha / 4) * dataframe['hl2'] + .5 * alpha * alpha * dataframe['hl2'].shift(
        1) - (alpha - .75 * alpha * alpha) * dataframe['hl2'].shift(2) + 2 * (
                               1 - alpha) * dataframe['itt'].shift(1) - (1 - alpha) * (1 - alpha) * dataframe[
                           'itt'].shift(2)

    dataframe['signal'] = 2 * dataframe['itt'] - dataframe['itt'].shift(2)

    return dataframe['itt'], dataframe['signal']
