#!/usr/bin/env python3
"""
    Import Modules
"""
#from matplotlib import cm # was used to get other colour maps
# config and os modules
import librosa
from pyqtgraph.colormap import ColorMap
import config
import audio_handler as audio
import os
import sys
import ctypes

# math, time modules
import math
import time
import datetime
import numpy as np

# import pyqt5 modules
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer#QSize, QTimer, 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QTextCursor, QPalette

# use pyqtgraph instead of matplotlib
import pyqtgraph as pg

# import for debugging
import logging

# configure the logging
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO)

# application version
__version__ = '0.0.1'

"""
    PyQT5 Application
"""

# stores the python output into the array
global PYTHON_OUTPUT
PYTHON_OUTPUT = []

class MyStream(object):
    """my stream class used to """

    def __init__(self, boolean = False) -> None:
        super().__init__()
        self.displaying_error = boolean # determines if we display in red

    def write(self, text):

        global PYTHON_OUTPUT

        # if the text is not empty
        if len(text) > 0:

            # get the current lines
            lines = len(PYTHON_OUTPUT)

            # append str text
            PYTHON_OUTPUT.append(str(text))

            # if the lines exceed buffer limit then delete the first element always
            if lines >= 1500:
                # forget the first 500 values
                PYTHON_OUTPUT = PYTHON_OUTPUT[500:]

class Base(QMainWindow):

    def __init__(self): # initialization function
        super().__init__() # invoke derived class constructor
        # self.last_closes_size = 0 # used to gate keep update timer from spamming/wasting cpu usage on updating plots
        self.last_len_terminal_lines = 0 # also used to gate keep update timer from spamming wasting cpu usage
        self.setupUI() # invoke the ui initialize

    def force_buy(self):
        """invoked on GUI force buy button"""
        print("button00 was invoked!")

        # try:

        #     if not in_position and not busy:

        #         result = QMessageBox.question(self,
        #                 "Confirm Buy Override...",
        #                 "Are you sure you want force buy?",
        #                 QMessageBox.Yes| QMessageBox.No)

        #         if result == QMessageBox.Yes: # prevent miss clicks

        #             print("BUY FORCE WAS CONFIRMED")

        #             perform_indicator_calculations()

        #             # update stop loss/stop profit
        #             last_stop_loss = stop_loss_price
        #             last_stop_profit = stop_profit_price

        #             # create a new thread to force in position
        #             thread_place_buy_order = threading.Thread(target=t_force_position)
        #             thread_place_buy_order.start()
                
        #     else:

        #         response = QMessageBox.information(self, "Request Denied", "We are already in position, or the bot is busy!", QMessageBox.Ok)

        # except Exception as e:
        #     print(f"[FORCE BUY ERROR] {str(e)}")

    def force_sell(self):
        """invoked on GUI force sell button"""
        print("button01 was invoked!")

        # try:

        #     if in_position and not busy:

        #         result = QMessageBox.question(
        #             self,
        #             "Confirm Sell Override...",
        #             "Are you sure you want force sell?",
        #             QMessageBox.Yes| QMessageBox.No
        #         )

        #         if result == QMessageBox.Yes: # prevent miss clicks

        #             print("SELL FORCE WAS CONFIRMED")

        #             # create a new thread to force exit position
        #             thread_place_sell_order = threading.Thread(target=t_force_exit_position)
        #             thread_place_sell_order.start()
                
        #     else:

        #         response = QMessageBox.information(self, "Request Denied", "We are not in position to sell, or the bot is busy!", QMessageBox.Ok)

        # except Exception as e:
        #     logging.critical( f"force_sell() -> invoked error: {str(e)}" )

    def indicator_text(self):
        """invoked by update plot to give status update of the bots"""

        # profit_summary = (f"PROFIT/LOSS TOTAL: <font color=\'green\'>{str(CURRENCY_SYMBOL)}" if gains_loss_total > 0 else f"PROFIT/LOSS TOTAL: <font color=\'red\'>{str(CURRENCY_SYMBOL)}") + f"{str(round(gains_loss_total,CURRENCY_ROUNDING))}</font>"
        # trend = str(TRADE_TREND_IS_BULLISH).lower().replace('1','<font color=\'green\'>bullish</font>').replace('0','<font color=\'red\'>bearish</font>')

        # in_position_str = (f"<font color=\'green\'>YES</font> | SL: <font color=\'#ff1515\'>{round(last_stop_loss,TRADE_ASSET_PRECISION)}:{round(last_stop_loss_limit,TRADE_ASSET_PRECISION)}</font> | HODL:<font color=\'#7d7dff\'>{round(last_buy_in_dict['avg'],TRADE_ASSET_PRECISION)}</font> | SP: <font color=\'#0d98ba\'>{round(last_stop_profit,TRADE_ASSET_PRECISION)}:{round(last_stop_profit_limit,TRADE_ASSET_PRECISION)}</font>" if in_position else "<font color=\'red\'>NO</font>")

        # if TRADE_WITH_BLANKS:
        #     profit_summary += " | <font color=\'orange\'>DEBUG</font>"

        s = ""

        if self.audio_handler is None:
            s = "<font color=\'red\'>NO AUDIO HANDLER</font>"
        else:
            s = (f"sample rate: <font color=\'orange\'>{self.audio_handler.SAMPLE_RATE}</font> Hz | channels: <font color=\'cyan\'>{self.audio_handler.CHANNELS}</font> | chunk: <font color=\'lime\'>{self.audio_handler.CHUNK}</font>").upper()

        return s

    def setupUI(self):

        #self.setToolTip('This is a <b>QWidget</b> widget')

        # act_force_buy = QAction('BUTTON 00', self)
        # #startAct.setShortcut('Ctrl+Q')
        # act_force_buy.triggered.connect(self.force_buy)

        # act_force_sell = QAction('BUTTON 01', self)
        # #pauseAct.setShortcut('Ctrl+W')
        # act_force_sell.triggered.connect(self.force_sell)

        # self.toolbar = self.addToolBar('Override Controls Toolbar')
        # self.toolbar.addAction(act_force_buy)
        # self.toolbar.addAction(act_force_sell)

        """
            Initial Variables 
        """

        self.audio_handler = None # set to none so that we can check if it is None later
        self.source = [] # source of our frame buffer

        self.fft_data = [] # data for fft
        self.fft_freq = [] # frequency for fft

        # get the file directory
        root_dir_path = os.path.dirname(os.path.realpath(__file__))

        """
            label indicator
        """

        # label indicator
        self.indicator_text_label = QLabel()

        # update indicator text
        self.indicator_text_label.setText( self.indicator_text() )

        # console output
        self.textEdit = QTextEdit()
        self.textEdit.setReadOnly(True)
        self.textEdit.verticalScrollBar().setValue(1)

        """
            Pyqtplot render target
        """
        pg.setConfigOptions(antialias=True) # set antialiasing on for prettier plots
        self.pyqtplot_rendertarget = pg.GraphicsLayoutWidget(title="graphics window".upper())

        # has a list of all our plots that we will dynamically update
        self.traces = dict()

        """
            Canvas References for Plotting
        """
    
        self.amplitude_canvas = self.pyqtplot_rendertarget.addPlot(title="audio amplitude".upper(), row=0, col=0)
        self.fft_canvas = self.pyqtplot_rendertarget.addPlot(title="Fourier Wave Transform".upper(), row=1, col=0)
        self.spg_canvas = self.pyqtplot_rendertarget.addPlot(title="spectrogram".upper(), row=2, col=0)
        self.mel_spec_canvas = self.pyqtplot_rendertarget.addPlot(title="mel spectrogram".upper(), row=3, col=0)
        
        # plasma colour map from matplotlib without importing it
        colourmap_lut = np.asarray([
            [5.03830e-02, 2.98030e-02, 5.27975e-01, 1.00000e+00],
            [6.35360e-02, 2.84260e-02, 5.33124e-01, 1.00000e+00],
            [7.53530e-02, 2.72060e-02, 5.38007e-01, 1.00000e+00],
            [8.62220e-02, 2.61250e-02, 5.42658e-01, 1.00000e+00],
            [9.63790e-02, 2.51650e-02, 5.47103e-01, 1.00000e+00],
            [1.05980e-01, 2.43090e-02, 5.51368e-01, 1.00000e+00],
            [1.15124e-01, 2.35560e-02, 5.55468e-01, 1.00000e+00],
            [1.23903e-01, 2.28780e-02, 5.59423e-01, 1.00000e+00],
            [1.32381e-01, 2.22580e-02, 5.63250e-01, 1.00000e+00],
            [1.40603e-01, 2.16870e-02, 5.66959e-01, 1.00000e+00],
            [1.48607e-01, 2.11540e-02, 5.70562e-01, 1.00000e+00],
            [1.56421e-01, 2.06510e-02, 5.74065e-01, 1.00000e+00],
            [1.64070e-01, 2.01710e-02, 5.77478e-01, 1.00000e+00],
            [1.71574e-01, 1.97060e-02, 5.80806e-01, 1.00000e+00],
            [1.78950e-01, 1.92520e-02, 5.84054e-01, 1.00000e+00],
            [1.86213e-01, 1.88030e-02, 5.87228e-01, 1.00000e+00],
            [1.93374e-01, 1.83540e-02, 5.90330e-01, 1.00000e+00],
            [2.00445e-01, 1.79020e-02, 5.93364e-01, 1.00000e+00],
            [2.07435e-01, 1.74420e-02, 5.96333e-01, 1.00000e+00],
            [2.14350e-01, 1.69730e-02, 5.99239e-01, 1.00000e+00],
            [2.21197e-01, 1.64970e-02, 6.02083e-01, 1.00000e+00],
            [2.27983e-01, 1.60070e-02, 6.04867e-01, 1.00000e+00],
            [2.34715e-01, 1.55020e-02, 6.07592e-01, 1.00000e+00],
            [2.41396e-01, 1.49790e-02, 6.10259e-01, 1.00000e+00],
            [2.48032e-01, 1.44390e-02, 6.12868e-01, 1.00000e+00],
            [2.54627e-01, 1.38820e-02, 6.15419e-01, 1.00000e+00],
            [2.61183e-01, 1.33080e-02, 6.17911e-01, 1.00000e+00],
            [2.67703e-01, 1.27160e-02, 6.20346e-01, 1.00000e+00],
            [2.74191e-01, 1.21090e-02, 6.22722e-01, 1.00000e+00],
            [2.80648e-01, 1.14880e-02, 6.25038e-01, 1.00000e+00],
            [2.87076e-01, 1.08550e-02, 6.27295e-01, 1.00000e+00],
            [2.93478e-01, 1.02130e-02, 6.29490e-01, 1.00000e+00],
            [2.99855e-01, 9.56100e-03, 6.31624e-01, 1.00000e+00],
            [3.06210e-01, 8.90200e-03, 6.33694e-01, 1.00000e+00],
            [3.12543e-01, 8.23900e-03, 6.35700e-01, 1.00000e+00],
            [3.18856e-01, 7.57600e-03, 6.37640e-01, 1.00000e+00],
            [3.25150e-01, 6.91500e-03, 6.39512e-01, 1.00000e+00],
            [3.31426e-01, 6.26100e-03, 6.41316e-01, 1.00000e+00],
            [3.37683e-01, 5.61800e-03, 6.43049e-01, 1.00000e+00],
            [3.43925e-01, 4.99100e-03, 6.44710e-01, 1.00000e+00],
            [3.50150e-01, 4.38200e-03, 6.46298e-01, 1.00000e+00],
            [3.56359e-01, 3.79800e-03, 6.47810e-01, 1.00000e+00],
            [3.62553e-01, 3.24300e-03, 6.49245e-01, 1.00000e+00],
            [3.68733e-01, 2.72400e-03, 6.50601e-01, 1.00000e+00],
            [3.74897e-01, 2.24500e-03, 6.51876e-01, 1.00000e+00],
            [3.81047e-01, 1.81400e-03, 6.53068e-01, 1.00000e+00],
            [3.87183e-01, 1.43400e-03, 6.54177e-01, 1.00000e+00],
            [3.93304e-01, 1.11400e-03, 6.55199e-01, 1.00000e+00],
            [3.99411e-01, 8.59000e-04, 6.56133e-01, 1.00000e+00],
            [4.05503e-01, 6.78000e-04, 6.56977e-01, 1.00000e+00],
            [4.11580e-01, 5.77000e-04, 6.57730e-01, 1.00000e+00],
            [4.17642e-01, 5.64000e-04, 6.58390e-01, 1.00000e+00],
            [4.23689e-01, 6.46000e-04, 6.58956e-01, 1.00000e+00],
            [4.29719e-01, 8.31000e-04, 6.59425e-01, 1.00000e+00],
            [4.35734e-01, 1.12700e-03, 6.59797e-01, 1.00000e+00],
            [4.41732e-01, 1.54000e-03, 6.60069e-01, 1.00000e+00],
            [4.47714e-01, 2.08000e-03, 6.60240e-01, 1.00000e+00],
            [4.53677e-01, 2.75500e-03, 6.60310e-01, 1.00000e+00],
            [4.59623e-01, 3.57400e-03, 6.60277e-01, 1.00000e+00],
            [4.65550e-01, 4.54500e-03, 6.60139e-01, 1.00000e+00],
            [4.71457e-01, 5.67800e-03, 6.59897e-01, 1.00000e+00],
            [4.77344e-01, 6.98000e-03, 6.59549e-01, 1.00000e+00],
            [4.83210e-01, 8.46000e-03, 6.59095e-01, 1.00000e+00],
            [4.89055e-01, 1.01270e-02, 6.58534e-01, 1.00000e+00],
            [4.94877e-01, 1.19900e-02, 6.57865e-01, 1.00000e+00],
            [5.00678e-01, 1.40550e-02, 6.57088e-01, 1.00000e+00],
            [5.06454e-01, 1.63330e-02, 6.56202e-01, 1.00000e+00],
            [5.12206e-01, 1.88330e-02, 6.55209e-01, 1.00000e+00],
            [5.17933e-01, 2.15630e-02, 6.54109e-01, 1.00000e+00],
            [5.23633e-01, 2.45320e-02, 6.52901e-01, 1.00000e+00],
            [5.29306e-01, 2.77470e-02, 6.51586e-01, 1.00000e+00],
            [5.34952e-01, 3.12170e-02, 6.50165e-01, 1.00000e+00],
            [5.40570e-01, 3.49500e-02, 6.48640e-01, 1.00000e+00],
            [5.46157e-01, 3.89540e-02, 6.47010e-01, 1.00000e+00],
            [5.51715e-01, 4.31360e-02, 6.45277e-01, 1.00000e+00],
            [5.57243e-01, 4.73310e-02, 6.43443e-01, 1.00000e+00],
            [5.62738e-01, 5.15450e-02, 6.41509e-01, 1.00000e+00],
            [5.68201e-01, 5.57780e-02, 6.39477e-01, 1.00000e+00],
            [5.73632e-01, 6.00280e-02, 6.37349e-01, 1.00000e+00],
            [5.79029e-01, 6.42960e-02, 6.35126e-01, 1.00000e+00],
            [5.84391e-01, 6.85790e-02, 6.32812e-01, 1.00000e+00],
            [5.89719e-01, 7.28780e-02, 6.30408e-01, 1.00000e+00],
            [5.95011e-01, 7.71900e-02, 6.27917e-01, 1.00000e+00],
            [6.00266e-01, 8.15160e-02, 6.25342e-01, 1.00000e+00],
            [6.05485e-01, 8.58540e-02, 6.22686e-01, 1.00000e+00],
            [6.10667e-01, 9.02040e-02, 6.19951e-01, 1.00000e+00],
            [6.15812e-01, 9.45640e-02, 6.17140e-01, 1.00000e+00],
            [6.20919e-01, 9.89340e-02, 6.14257e-01, 1.00000e+00],
            [6.25987e-01, 1.03312e-01, 6.11305e-01, 1.00000e+00],
            [6.31017e-01, 1.07699e-01, 6.08287e-01, 1.00000e+00],
            [6.36008e-01, 1.12092e-01, 6.05205e-01, 1.00000e+00],
            [6.40959e-01, 1.16492e-01, 6.02065e-01, 1.00000e+00],
            [6.45872e-01, 1.20898e-01, 5.98867e-01, 1.00000e+00],
            [6.50746e-01, 1.25309e-01, 5.95617e-01, 1.00000e+00],
            [6.55580e-01, 1.29725e-01, 5.92317e-01, 1.00000e+00],
            [6.60374e-01, 1.34144e-01, 5.88971e-01, 1.00000e+00],
            [6.65129e-01, 1.38566e-01, 5.85582e-01, 1.00000e+00],
            [6.69845e-01, 1.42992e-01, 5.82154e-01, 1.00000e+00],
            [6.74522e-01, 1.47419e-01, 5.78688e-01, 1.00000e+00],
            [6.79160e-01, 1.51848e-01, 5.75189e-01, 1.00000e+00],
            [6.83758e-01, 1.56278e-01, 5.71660e-01, 1.00000e+00],
            [6.88318e-01, 1.60709e-01, 5.68103e-01, 1.00000e+00],
            [6.92840e-01, 1.65141e-01, 5.64522e-01, 1.00000e+00],
            [6.97324e-01, 1.69573e-01, 5.60919e-01, 1.00000e+00],
            [7.01769e-01, 1.74005e-01, 5.57296e-01, 1.00000e+00],
            [7.06178e-01, 1.78437e-01, 5.53657e-01, 1.00000e+00],
            [7.10549e-01, 1.82868e-01, 5.50004e-01, 1.00000e+00],
            [7.14883e-01, 1.87299e-01, 5.46338e-01, 1.00000e+00],
            [7.19181e-01, 1.91729e-01, 5.42663e-01, 1.00000e+00],
            [7.23444e-01, 1.96158e-01, 5.38981e-01, 1.00000e+00],
            [7.27670e-01, 2.00586e-01, 5.35293e-01, 1.00000e+00],
            [7.31862e-01, 2.05013e-01, 5.31601e-01, 1.00000e+00],
            [7.36019e-01, 2.09439e-01, 5.27908e-01, 1.00000e+00],
            [7.40143e-01, 2.13864e-01, 5.24216e-01, 1.00000e+00],
            [7.44232e-01, 2.18288e-01, 5.20524e-01, 1.00000e+00],
            [7.48289e-01, 2.22711e-01, 5.16834e-01, 1.00000e+00],
            [7.52312e-01, 2.27133e-01, 5.13149e-01, 1.00000e+00],
            [7.56304e-01, 2.31555e-01, 5.09468e-01, 1.00000e+00],
            [7.60264e-01, 2.35976e-01, 5.05794e-01, 1.00000e+00],
            [7.64193e-01, 2.40396e-01, 5.02126e-01, 1.00000e+00],
            [7.68090e-01, 2.44817e-01, 4.98465e-01, 1.00000e+00],
            [7.71958e-01, 2.49237e-01, 4.94813e-01, 1.00000e+00],
            [7.75796e-01, 2.53658e-01, 4.91171e-01, 1.00000e+00],
            [7.79604e-01, 2.58078e-01, 4.87539e-01, 1.00000e+00],
            [7.83383e-01, 2.62500e-01, 4.83918e-01, 1.00000e+00],
            [7.87133e-01, 2.66922e-01, 4.80307e-01, 1.00000e+00],
            [7.90855e-01, 2.71345e-01, 4.76706e-01, 1.00000e+00],
            [7.94549e-01, 2.75770e-01, 4.73117e-01, 1.00000e+00],
            [7.98216e-01, 2.80197e-01, 4.69538e-01, 1.00000e+00],
            [8.01855e-01, 2.84626e-01, 4.65971e-01, 1.00000e+00],
            [8.05467e-01, 2.89057e-01, 4.62415e-01, 1.00000e+00],
            [8.09052e-01, 2.93491e-01, 4.58870e-01, 1.00000e+00],
            [8.12612e-01, 2.97928e-01, 4.55338e-01, 1.00000e+00],
            [8.16144e-01, 3.02368e-01, 4.51816e-01, 1.00000e+00],
            [8.19651e-01, 3.06812e-01, 4.48306e-01, 1.00000e+00],
            [8.23132e-01, 3.11261e-01, 4.44806e-01, 1.00000e+00],
            [8.26588e-01, 3.15714e-01, 4.41316e-01, 1.00000e+00],
            [8.30018e-01, 3.20172e-01, 4.37836e-01, 1.00000e+00],
            [8.33422e-01, 3.24635e-01, 4.34366e-01, 1.00000e+00],
            [8.36801e-01, 3.29105e-01, 4.30905e-01, 1.00000e+00],
            [8.40155e-01, 3.33580e-01, 4.27455e-01, 1.00000e+00],
            [8.43484e-01, 3.38062e-01, 4.24013e-01, 1.00000e+00],
            [8.46788e-01, 3.42551e-01, 4.20579e-01, 1.00000e+00],
            [8.50066e-01, 3.47048e-01, 4.17153e-01, 1.00000e+00],
            [8.53319e-01, 3.51553e-01, 4.13734e-01, 1.00000e+00],
            [8.56547e-01, 3.56066e-01, 4.10322e-01, 1.00000e+00],
            [8.59750e-01, 3.60588e-01, 4.06917e-01, 1.00000e+00],
            [8.62927e-01, 3.65119e-01, 4.03519e-01, 1.00000e+00],
            [8.66078e-01, 3.69660e-01, 4.00126e-01, 1.00000e+00],
            [8.69203e-01, 3.74212e-01, 3.96738e-01, 1.00000e+00],
            [8.72303e-01, 3.78774e-01, 3.93355e-01, 1.00000e+00],
            [8.75376e-01, 3.83347e-01, 3.89976e-01, 1.00000e+00],
            [8.78423e-01, 3.87932e-01, 3.86600e-01, 1.00000e+00],
            [8.81443e-01, 3.92529e-01, 3.83229e-01, 1.00000e+00],
            [8.84436e-01, 3.97139e-01, 3.79860e-01, 1.00000e+00],
            [8.87402e-01, 4.01762e-01, 3.76494e-01, 1.00000e+00],
            [8.90340e-01, 4.06398e-01, 3.73130e-01, 1.00000e+00],
            [8.93250e-01, 4.11048e-01, 3.69768e-01, 1.00000e+00],
            [8.96131e-01, 4.15712e-01, 3.66407e-01, 1.00000e+00],
            [8.98984e-01, 4.20392e-01, 3.63047e-01, 1.00000e+00],
            [9.01807e-01, 4.25087e-01, 3.59688e-01, 1.00000e+00],
            [9.04601e-01, 4.29797e-01, 3.56329e-01, 1.00000e+00],
            [9.07365e-01, 4.34524e-01, 3.52970e-01, 1.00000e+00],
            [9.10098e-01, 4.39268e-01, 3.49610e-01, 1.00000e+00],
            [9.12800e-01, 4.44029e-01, 3.46251e-01, 1.00000e+00],
            [9.15471e-01, 4.48807e-01, 3.42890e-01, 1.00000e+00],
            [9.18109e-01, 4.53603e-01, 3.39529e-01, 1.00000e+00],
            [9.20714e-01, 4.58417e-01, 3.36166e-01, 1.00000e+00],
            [9.23287e-01, 4.63251e-01, 3.32801e-01, 1.00000e+00],
            [9.25825e-01, 4.68103e-01, 3.29435e-01, 1.00000e+00],
            [9.28329e-01, 4.72975e-01, 3.26067e-01, 1.00000e+00],
            [9.30798e-01, 4.77867e-01, 3.22697e-01, 1.00000e+00],
            [9.33232e-01, 4.82780e-01, 3.19325e-01, 1.00000e+00],
            [9.35630e-01, 4.87712e-01, 3.15952e-01, 1.00000e+00],
            [9.37990e-01, 4.92667e-01, 3.12575e-01, 1.00000e+00],
            [9.40313e-01, 4.97642e-01, 3.09197e-01, 1.00000e+00],
            [9.42598e-01, 5.02639e-01, 3.05816e-01, 1.00000e+00],
            [9.44844e-01, 5.07658e-01, 3.02433e-01, 1.00000e+00],
            [9.47051e-01, 5.12699e-01, 2.99049e-01, 1.00000e+00],
            [9.49217e-01, 5.17763e-01, 2.95662e-01, 1.00000e+00],
            [9.51344e-01, 5.22850e-01, 2.92275e-01, 1.00000e+00],
            [9.53428e-01, 5.27960e-01, 2.88883e-01, 1.00000e+00],
            [9.55470e-01, 5.33093e-01, 2.85490e-01, 1.00000e+00],
            [9.57469e-01, 5.38250e-01, 2.82096e-01, 1.00000e+00],
            [9.59424e-01, 5.43431e-01, 2.78701e-01, 1.00000e+00],
            [9.61336e-01, 5.48636e-01, 2.75305e-01, 1.00000e+00],
            [9.63203e-01, 5.53865e-01, 2.71909e-01, 1.00000e+00],
            [9.65024e-01, 5.59118e-01, 2.68513e-01, 1.00000e+00],
            [9.66798e-01, 5.64396e-01, 2.65118e-01, 1.00000e+00],
            [9.68526e-01, 5.69700e-01, 2.61721e-01, 1.00000e+00],
            [9.70205e-01, 5.75028e-01, 2.58325e-01, 1.00000e+00],
            [9.71835e-01, 5.80382e-01, 2.54931e-01, 1.00000e+00],
            [9.73416e-01, 5.85761e-01, 2.51540e-01, 1.00000e+00],
            [9.74947e-01, 5.91165e-01, 2.48151e-01, 1.00000e+00],
            [9.76428e-01, 5.96595e-01, 2.44767e-01, 1.00000e+00],
            [9.77856e-01, 6.02051e-01, 2.41387e-01, 1.00000e+00],
            [9.79233e-01, 6.07532e-01, 2.38013e-01, 1.00000e+00],
            [9.80556e-01, 6.13039e-01, 2.34646e-01, 1.00000e+00],
            [9.81826e-01, 6.18572e-01, 2.31287e-01, 1.00000e+00],
            [9.83041e-01, 6.24131e-01, 2.27937e-01, 1.00000e+00],
            [9.84199e-01, 6.29718e-01, 2.24595e-01, 1.00000e+00],
            [9.85301e-01, 6.35330e-01, 2.21265e-01, 1.00000e+00],
            [9.86345e-01, 6.40969e-01, 2.17948e-01, 1.00000e+00],
            [9.87332e-01, 6.46633e-01, 2.14648e-01, 1.00000e+00],
            [9.88260e-01, 6.52325e-01, 2.11364e-01, 1.00000e+00],
            [9.89128e-01, 6.58043e-01, 2.08100e-01, 1.00000e+00],
            [9.89935e-01, 6.63787e-01, 2.04859e-01, 1.00000e+00],
            [9.90681e-01, 6.69558e-01, 2.01642e-01, 1.00000e+00],
            [9.91365e-01, 6.75355e-01, 1.98453e-01, 1.00000e+00],
            [9.91985e-01, 6.81179e-01, 1.95295e-01, 1.00000e+00],
            [9.92541e-01, 6.87030e-01, 1.92170e-01, 1.00000e+00],
            [9.93032e-01, 6.92907e-01, 1.89084e-01, 1.00000e+00],
            [9.93456e-01, 6.98810e-01, 1.86041e-01, 1.00000e+00],
            [9.93814e-01, 7.04741e-01, 1.83043e-01, 1.00000e+00],
            [9.94103e-01, 7.10698e-01, 1.80097e-01, 1.00000e+00],
            [9.94324e-01, 7.16681e-01, 1.77208e-01, 1.00000e+00],
            [9.94474e-01, 7.22691e-01, 1.74381e-01, 1.00000e+00],
            [9.94553e-01, 7.28728e-01, 1.71622e-01, 1.00000e+00],
            [9.94561e-01, 7.34791e-01, 1.68938e-01, 1.00000e+00],
            [9.94495e-01, 7.40880e-01, 1.66335e-01, 1.00000e+00],
            [9.94355e-01, 7.46995e-01, 1.63821e-01, 1.00000e+00],
            [9.94141e-01, 7.53137e-01, 1.61404e-01, 1.00000e+00],
            [9.93851e-01, 7.59304e-01, 1.59092e-01, 1.00000e+00],
            [9.93482e-01, 7.65499e-01, 1.56891e-01, 1.00000e+00],
            [9.93033e-01, 7.71720e-01, 1.54808e-01, 1.00000e+00],
            [9.92505e-01, 7.77967e-01, 1.52855e-01, 1.00000e+00],
            [9.91897e-01, 7.84239e-01, 1.51042e-01, 1.00000e+00],
            [9.91209e-01, 7.90537e-01, 1.49377e-01, 1.00000e+00],
            [9.90439e-01, 7.96859e-01, 1.47870e-01, 1.00000e+00],
            [9.89587e-01, 8.03205e-01, 1.46529e-01, 1.00000e+00],
            [9.88648e-01, 8.09579e-01, 1.45357e-01, 1.00000e+00],
            [9.87621e-01, 8.15978e-01, 1.44363e-01, 1.00000e+00],
            [9.86509e-01, 8.22401e-01, 1.43557e-01, 1.00000e+00],
            [9.85314e-01, 8.28846e-01, 1.42945e-01, 1.00000e+00],
            [9.84031e-01, 8.35315e-01, 1.42528e-01, 1.00000e+00],
            [9.82653e-01, 8.41812e-01, 1.42303e-01, 1.00000e+00],
            [9.81190e-01, 8.48329e-01, 1.42279e-01, 1.00000e+00],
            [9.79644e-01, 8.54866e-01, 1.42453e-01, 1.00000e+00],
            [9.77995e-01, 8.61432e-01, 1.42808e-01, 1.00000e+00],
            [9.76265e-01, 8.68016e-01, 1.43351e-01, 1.00000e+00],
            [9.74443e-01, 8.74622e-01, 1.44061e-01, 1.00000e+00],
            [9.72530e-01, 8.81250e-01, 1.44923e-01, 1.00000e+00],
            [9.70533e-01, 8.87896e-01, 1.45919e-01, 1.00000e+00],
            [9.68443e-01, 8.94564e-01, 1.47014e-01, 1.00000e+00],
            [9.66271e-01, 9.01249e-01, 1.48180e-01, 1.00000e+00],
            [9.64021e-01, 9.07950e-01, 1.49370e-01, 1.00000e+00],
            [9.61681e-01, 9.14672e-01, 1.50520e-01, 1.00000e+00],
            [9.59276e-01, 9.21407e-01, 1.51566e-01, 1.00000e+00],
            [9.56808e-01, 9.28152e-01, 1.52409e-01, 1.00000e+00],
            [9.54287e-01, 9.34908e-01, 1.52921e-01, 1.00000e+00],
            [9.51726e-01, 9.41671e-01, 1.52925e-01, 1.00000e+00],
            [9.49151e-01, 9.48435e-01, 1.52178e-01, 1.00000e+00],
            [9.46602e-01, 9.55190e-01, 1.50328e-01, 1.00000e+00],
            [9.44152e-01, 9.61916e-01, 1.46861e-01, 1.00000e+00],
            [9.41896e-01, 9.68590e-01, 1.40956e-01, 1.00000e+00],
            [9.40015e-01, 9.75158e-01, 1.31326e-01, 1.00000e+00],
            [5.03830e-02, 2.98030e-02, 5.27975e-01, 1.00000e+00],
            [9.40015e-01, 9.75158e-01, 1.31326e-01, 1.00000e+00],
            [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00]
            ])
        #print(colormap._lut)
        #print(colormap.N)
        # Convert matplotlib colormap from 0-1 to 0-255 for PyQtGraph
        #lut = (colormap._lut * 255).view(np.ndarray)[:colormap.N] 
        lut = (colourmap_lut * 255).view(np.ndarray)[:256] 

        # image for spectrogram
        self.spectrogram_img = pg.ImageItem()
        # add the spectrogram_img to the spg canvas
        self.spg_canvas.addItem(self.spectrogram_img)
        # set the image array to zeros
        self.spectrogram_img_array = np.zeros((1000, int(config.CHUNK_SIZE/2+1)))

        # set colormap
        self.spectrogram_img.setLookupTable(lut)
        self.spectrogram_img.setLevels([-50,40])

        # setup the correct scaling for y-axis
        freq = np.arange(0,int(config.SAMPLE_RATE/2))
        #freq = np.arange((config.CHUNK_SIZE/2)+1)/(float(config.CHUNK_SIZE)/config.SAMPLE_RATE)
        
        # set the y-axis to the correct frequency scale
        yscale = 1.0/(self.spectrogram_img_array.shape[1]/freq[-1])

        # set spectrogram_img scale
        self.spectrogram_img.scale((1./config.SAMPLE_RATE)*config.CHUNK_SIZE, yscale)

        # set the label of the canvas to show frequency on the Y axis
        self.spg_canvas.setLabel('left', 'Frequency', units='Hz')

        """
            Mel Spectrogram plot properties and setup
        """

        # image for mel spectrogram
        self.mel_spectrogram_img = pg.ImageItem()
        # add the mel_spectrogram_img to the mel_spec canvas
        self.mel_spec_canvas.addItem(self.mel_spectrogram_img)
        # set the image array to zeros
        self.mel_spectrogram_img_array = np.zeros((1000, int(config.CHUNK_SIZE/2+1)))

        # set colormap
        self.mel_spectrogram_img.setLookupTable(lut)
        self.mel_spectrogram_img.setLevels([-50,40])

        # setup the correct scaling for y-axis
        freq = np.arange(0,int(config.SAMPLE_RATE/2))
        #freq = np.arange((config.CHUNK_SIZE/2)+1)/(float(config.CHUNK_SIZE)/config.SAMPLE_RATE)
        
        # set the y-axis to the correct frequency scale
        yscale = 1.0/(self.mel_spectrogram_img_array.shape[1]/freq[-1])

        # set spectrogram_img scale
        self.mel_spectrogram_img.scale((1./config.SAMPLE_RATE)*config.CHUNK_SIZE, yscale)

        # set the label of the canvas to show frequency on the Y axis
        self.mel_spec_canvas.setLabel('left', 'Frequency', units='Hz')


        # bipolar colormap
        # pos = np.array([0., 1., 0.5, 0.25, 0.75])
        # color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        
        #cmap = pg.ColorMap(pos, lut_cubehelix) # color
        #lut = cmap.getLookupTable(0.0, 1.0, 256)

        # colormap = cm.get_cmap("plasma")
        # colormap._init()
        # np.set_printoptions(threshold=sys.maxsize)



        # prepare window for later use
        #self.win = np.hanning(config.CHUNK_SIZE)


        """
            Setup, GUI layout
        """

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(500) # 500
        self.timer.timeout.connect(self.update_textedit) # self.update_plot

        # Setup a timer to trigger the redraw by calling update_plot.
        self.pre_process_timer = QTimer()
        self.pre_process_timer.setInterval(0) # 500
        self.pre_process_timer.timeout.connect(self.pre_process_data) # self.update_plot

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer2 = QTimer()
        self.timer2.setInterval(0) # 3000
        self.timer2.timeout.connect(self.update_plot) # self.update_plot

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        Overall_Layout = QGridLayout(central_widget)

        #grid_layout = QGridLayout()
        #self.setLayout(grid_layout)

        Overall_Layout.setRowStretch(3, 1)
        #Overall_Layout.setColumnStretch(1, 2)
        Overall_Layout.addWidget( self.indicator_text_label, 1, 1 )
        Overall_Layout.addWidget( self.textEdit, 2, 1 )
        Overall_Layout.addWidget( self.pyqtplot_rendertarget, 3, 1 )

        # Overall_Layout.addWidget( self.canvas, 3, 1 )
        # Overall_Layout.addWidget( self.fft_canvas, 4, 1 )
        # Overall_Layout.addWidget( self.spg_canvas, 4, 1 )

        self.setGeometry(300, 300, 800, 900)        # set the size of the window
        self.setWindowTitle('multi-label sound event classification system'.upper())              # set window title Audio.To.SpectroGraph
        self.setWindowIcon(QIcon(os.path.join( root_dir_path, 'icon.png' )))       # set window icon

        # display the window
        self.show()

        # hook events for python program out so that we can view debug information of the last 250 characters
        sys.stdout = MyStream()
        sys.stderr = MyStream(True)


        # update text timer
        self.timer.start()

        # attempt to initialize the audio handler object and start the audio stream
        try:
            self.audio_handler = audio.AudioHandler()
            self.audio_handler.start()
        except Exception as e:
            print(f"Error when initializing and starting audio handler {e}")
            logging.critical("Error: %s" % e)
        else:
            if self.audio_handler is not None:
                if self.audio_handler.is_stream_active():
                    print("Audio stream is active")


                    # begin bot updating on another thread
                    # self.thread_audio_data_fetcher = threading.Thread(target=self.frame_fetcher)
                    # self.thread_audio_data_fetcher.start()

                # only big updating plots if we have an audio stream duh!

                # begin the pre-process data timer
                self.pre_process_timer.start()

                # update plots
                self.update_plot()

                # update plot timer
                self.timer2.start()
            else:
                print("Audio stream is not active")

    def set_plotdata(self, name, data_x, data_y, auto_scale=True):
        if name in self.traces:
            #if np.array_equal(data_x,data_y):
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'amplitude':
                self.traces[name] = self.amplitude_canvas.plot(pen='c', width=3)
                self.amplitude_canvas.setYRange(-2.5, 2.5, padding=0)
                self.amplitude_canvas.setXRange(0, len(self.source), padding=0.005)

                if auto_scale:
                    self.amplitude_canvas.setMouseEnabled(x=False,y=False)  
                    self.amplitude_canvas.enableAutoRange(axis='y', enable=True)
                    self.amplitude_canvas.setAutoVisible(y=1.0)  
                    self.amplitude_canvas.setAspectLocked(lock=False)  


            elif name == 'amplitude2':
                self.traces[name] = self.amplitude_canvas.plot(pen='c', width=3)
                self.amplitude_canvas.setYRange(-2.5, 2.5, padding=0)
                self.amplitude_canvas.setXRange(0, len(self.source), padding=0.005)
            
                if auto_scale:
                    self.amplitude_canvas.setMouseEnabled(x=False,y=False)  
                    self.amplitude_canvas.enableAutoRange(axis='y', enable=True)
                    self.amplitude_canvas.setAutoVisible(y=1.0)  
                    self.amplitude_canvas.setAspectLocked(lock=False)  

            elif name == 'fft':
                self.traces[name] = self.fft_canvas.plot(pen=pg.mkPen({'color': "#ff2a00"}), width=3)
                self.fft_canvas.setYRange(0, 250, padding=0)
                self.fft_canvas.setXRange(0, int(self.audio_handler.SAMPLE_RATE/2), padding=0.005)
                
                if auto_scale:
                    self.fft_canvas.setMouseEnabled(x=False,y=False)  
                    self.fft_canvas.enableAutoRange(axis='y', enable=True)
                    self.fft_canvas.setAutoVisible(y=3.0)  
                    self.fft_canvas.setAspectLocked(lock=False)  
            
            elif name == "spectrogram":
                pass
            # if name == 'spectrum':
            #     self.traces[name] = self.spectrum.plot(pen='m', width=3)
            #     self.spectrum.setLogMode(x=True, y=True)
            #     self.spectrum.setYRange(-4, 0, padding=0)
            #     self.spectrum.setXRange(
            #         np.log10(20), np.log10(self.RATE / 2), padding=0.005)

    def pre_process_data(self):
        """data that is required by graphs but kept outside the update loop"""

        # we have to unify the source of the audio data,
        # because it handled in a separate thread and we need to synchronize it
        self.source = self.audio_handler.frame_buffer

        try:
            # we need to numpy abs, because FFT will show negative frequencies and amplitudes
            self.fft_data = np.abs(np.fft.fft(self.source[len(self.source)-1]))
            self.fft_freq = np.abs(np.fft.fftfreq(len(self.fft_data), 1.0/config.SAMPLE_RATE))#np.fft.fftfreq(len(source[len(source)-1]), 1.0/config.SAMPLE_RATE)
        except Exception as e:
            pass
            #print(f"Error when calculating fft data {e}")

        # make sure source is long enough
        if len(self.source) > 0:
            self.simplified_data = [] # our list of our simplified data
            self.negative_simplified_data = [] # our list of our simplified data

            # remove the unnecessary data only get the lowest and highest np
            for data in self.source:
                self.simplified_data.append(np.nanmax(data)) # [np.nanmin(data), np.nanmax(data)]
                self.negative_simplified_data.append(np.nanmin(data))

            # set the wave amplitude data
            self.wave_x = list(range(len(self.simplified_data)))#list(range(len(self.source)))
            self.wave_y = self.simplified_data
            self.wave_negative_y = self.negative_simplified_data
            
    def update_plot(self, override = False):
        """Triggered by a timer to invoke canvas to update and redraw."""
        
        if override:
            print("override was invoked force update_plot!")

        # TODO: migrate matplot to pyqtplot
        # Thanks! https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python/blob/master/audio_spectrumQT.py
        # Thanks! (Spectrogram Help) https://stackoverflow.com/questions/40374738/my-pyqt-plots-y-axes-are-upside-down-even-the-text

        # update indicator text
        self.indicator_text_label.setText( self.indicator_text() )

        if len(self.source) > 0:

            # update plots
            self.set_plotdata("amplitude", self.wave_x, self.wave_y )
            self.set_plotdata("amplitude2", self.wave_x, self.wave_negative_y )
            self.set_plotdata("fft", self.fft_freq, self.fft_data )

            # update spectrogram if the np buffer is greater than the chunk size
            # because if its too small its not enough for the spectrogram to render
            if len(self.audio_handler.np_buffer) > self.audio_handler.CHUNK and len(self.source) > 0:
                #np_array = librosa.feature.melspectrogram(y=self.audio_handler.np_buffer, sr=config.SAMPLE_RATE, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0)
                
                # normalized, windowed frequencies in data chunk
                spec = np.fft.rfft(self.source[len(self.source)-1])#self.audio_handler.np_buffer#np.fft.rfft(config.CHUNK_SIZE*self.win) / config.CHUNK_SIZE
                
                # get magnitude 
                psd = abs(spec)
                
                # convert to dB scale
                psd = 20 * np.log10(psd)

                # roll down one and replace leading edge with new data
                self.spectrogram_img_array = np.roll(self.spectrogram_img_array, -1, 0) # roll down one row
                self.spectrogram_img_array[-1:] = psd[0:self.spectrogram_img_array.shape[1]] # only take the first half of the spectrum
                self.spectrogram_img.setImage(self.spectrogram_img_array, autoLevels=False) # set the image data

                # # update the mel spectrogram
                # mel_spec = librosa.feature.melspectrogram(y=self.audio_handler.np_buffer, sr=config.SAMPLE_RATE, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0)

                # # convert to dB scale
                # mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                # #mel_spec = 20 * np.log10(mel_spec)

                # # roll down one and replace leading edge with new data
                # self.mel_spectrogram_img_array = np.roll(self.mel_spectrogram_img_array, -1, 0) # roll down one row
                # self.mel_spectrogram_img_array[-1:] = mel_spec[0:self.mel_spectrogram_img_array.shape[1]] # only take the first half of the spectrum
                # self.mel_spectrogram_img.setImage(self.mel_spectrogram_img_array, autoLevels=False) # set the image data

    def update_textedit(self):
        """Triggered by a timer to invoke an update on text edit"""

        global PYTHON_OUTPUT

        # only update on change
        if len(PYTHON_OUTPUT) != self.last_len_terminal_lines:

            # update last count
            self.last_len_terminal_lines = len(PYTHON_OUTPUT)

            self.textEdit.clear()

            full_string = ""
            for line in PYTHON_OUTPUT:
                full_string += line

            self.textEdit.insertPlainText( str(full_string) )
            self.textEdit.moveCursor(QTextCursor.End)

            # # update indicator text
            # self.indicator_text_label.setText( self.indicator_text() )

    def closeEvent(self,event):
        """close event is invoked on close but we want to prevent accidental close"""
        
        # query user if they want to close
        result = QMessageBox.question(self,
                      "Confirm Exit...",
                      "Are you sure you want to exit ?",
                      QMessageBox.Yes| QMessageBox.No)
        event.ignore()

        # if yes well then goodbye!
        if result == QMessageBox.Yes:

            # stop the audio stream before closing the application
            try:
                self.audio_handler.start()
            except Exception as e:
                print(f"Error stopping audio handler: {e}")
                logging.critical("Error stopping audio handler: %s" % e)

            sys.stdout = sys.__stdout__
            event.accept()


def execute():
    """starts the pyqt5 application"""

    print( "execute() -> on -> OS =", os.name )

    # detect if the app is running on the NT kernal if so we are on windows baby!
    if os.name.lower() == "nt":
        myappid = u'pyqt5.mlsecs.' + str(__version__) # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid) # adjust the app id of windows to the correct app id

    app = QApplication(sys.argv) # run the q application by passing in any system arguments through from the script
    
    # dark theme
    app.setStyle("Fusion")

    # override the pallet to darker
    palette = QPalette()
    palette.setColor(QPalette.Window, QtGui.QColor(53,53,53))
    palette.setColor(QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QPalette.Base, QtGui.QColor(15,15,15))
    palette.setColor(QPalette.AlternateBase, QtGui.QColor(53,53,53))
    palette.setColor(QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QPalette.Text, QtCore.Qt.white)
    palette.setColor(QPalette.Button, QtGui.QColor(53,53,53))
    palette.setColor(QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QPalette.BrightText, QtCore.Qt.red)
         
    palette.setColor(QPalette.Highlight, QtGui.QColor(142,45,197).lighter())
    palette.setColor(QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    baseObject = Base() # create a new instance of base class which will contrain our GUI information    
    sys.exit(app.exec_()) # exit the python code if the app has recieved the exit code status

"""
    Execution Entry Point
"""
if __name__ == '__main__':
    execute()