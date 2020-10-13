from ..node import Node


def RSI(node, period=14):
    '''Relative Strength Index

    Args:
        node (Node): input data
        period (int): RSI period
    '''
    raise NotImplementedError()


def MACD(node, period_fast=12, period_slow=26, signal=9):
    '''Moving Average Convergence/Divergence

    Args:
        node (Node): input data
        period_fast (int): Fast moving average period
        period_slow (int): Slow moving average period
        signal (int): MACD moving average period
    Returns:
        Node: node that emits tuple of (macd, macd_signal)
    '''
    raise NotImplementedError()


Node.rsi = RSI
Node.macd = MACD
