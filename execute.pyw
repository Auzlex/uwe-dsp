#!/usr/bin/env python3
"""
    Import Modules
"""
# config and os modules
import librosa
import config
import audio_handler as audio
import os
import sys
import ctypes
import threading

import struct

# math, time modules
import math
import time
import datetime
import numpy as np

# import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# configure the matplotlib to change the tick colours to white
plt.rcParams['xtick.color'] = "w"
plt.rcParams['ytick.color'] = "w"

# import pyqt5 modules
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer#QSize, QTimer, 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QTextCursor, QPalette

# configure matplotlib to use a QT backend
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# scipy fft
import scipy.fftpack

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

    def write(self, text):

        global PYTHON_OUTPUT

        if len(text) > 0:

            lines = len(PYTHON_OUTPUT)

            PYTHON_OUTPUT.append(str(text))

            # Add text to a QTextEdit...
            #self.qt_text_edit.insertPlainText( str(text) )
            #self.qt_text_edit.moveCursor(QTextCursor.End)

            # if the lines exceed buffer limit then delete the first element always
            if lines >= 1500:
                # forget the first 500 values
                PYTHON_OUTPUT = PYTHON_OUTPUT[500:]

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, titleinfo="N/A", width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi,facecolor="#1f2124")
        fig.suptitle( titleinfo, color="#ffffff" ) # set figure suptitle
        self.axes = fig.add_subplot(111)


        super(MplCanvas, self).__init__(fig)

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

        return (f"sample text update").upper()

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

        # get the file directory
        root_dir_path = os.path.dirname(os.path.realpath(__file__))

        # label indicator
        self.indicator_text_label = QLabel()
        self.indicator_text_label.setText( "sample text".upper() )

        # console output
        self.textEdit = QTextEdit()
        self.textEdit.setReadOnly(True)
        self.textEdit.verticalScrollBar().setValue(1)

        """
            Amplitiude canvas information
        """
        # setup canvas for candle information
        self.canvas = MplCanvas(self, titleinfo="audio amplitude".upper(), width=5, height=4, dpi=70)
        
        # sets the axis facecolour and label colour
        self.canvas.axes.set_facecolor('#23272a')
        self.canvas.axes.set_axisbelow(True)
        self.canvas.axes.tick_params(color="#1f2124", labelcolor='#ffffff')

        for spine in self.canvas.axes.spines.values():
            spine.set_edgecolor('#1d1e21')

        """
            Fourier Wave Transform canvas
        """
        # setup canvas for volatility detection with 
        self.fft_canvas = MplCanvas(self, titleinfo="Fourier Wave Transform".upper(), width=5, height=4, dpi=70)
        
        # sets the axis facecolour and label colour
        self.fft_canvas.axes.set_facecolor('#23272a')
        self.fft_canvas.axes.set_axisbelow(True)
        self.fft_canvas.axes.tick_params(color="#1f2124", labelcolor='#ffffff')

        #self.fft_canvas.axes.autoscale(False)

        for spine in self.fft_canvas.axes.spines.values():
            spine.set_edgecolor('#1d1e21')

        """
            Spectrograph Canvas
        """
        # setup canvas for volatility detection with 
        self.spg_canvas = MplCanvas(self, titleinfo="spectrograph".upper(), width=5, height=4, dpi=70)
        
        # sets the axis facecolour and label colour
        self.spg_canvas.axes.set_facecolor('#23272a')
        self.spg_canvas.axes.set_axisbelow(True)
        self.spg_canvas.axes.tick_params(color="#1f2124", labelcolor='#ffffff')

        for spine in self.spg_canvas.axes.spines.values():
            spine.set_edgecolor('#1d1e21')

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

        #Overall_Layout.setRowStretch(1, 1)
        #Overall_Layout.setColumnStretch(1, 2)
        Overall_Layout.addWidget( self.indicator_text_label, 1, 1 )
        Overall_Layout.addWidget( self.textEdit, 2, 1 )

        Overall_Layout.addWidget( self.canvas, 3, 1 )
        Overall_Layout.addWidget( self.fft_canvas, 4, 1 )
        Overall_Layout.addWidget( self.spg_canvas, 5, 1 )

        self.setGeometry(300, 300, 800, 900)        # set the size of the window
        self.setWindowTitle('multi-label sound event classification system'.upper())              # set window title Audio.To.SpectroGraph
        self.setWindowIcon(QIcon(os.path.join( root_dir_path, 'icon.png' )))       # set window icon

        # display the window
        self.show()

        # hook events for python program out so that we can view debug information of the last 250 characters
        sys.stdout = MyStream()
        #sys.stderr = MyStream()


        # update text timer
        self.timer.start()

        self.audio_handler = None # set to none so that we can check if it is None later
        self.source = [] # source of our frame buffer

        self.fft_data = [] # data for fft
        self.fft_freq = [] # frequency for fft

        self.rfft_source = []

        # draw canvas
        self.canvas.draw()              # draw the figure first
        self.fft_canvas.draw()   # draw the figure first
        self.spg_canvas.draw()          # draw the figure first
        
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
                # update plots
                self.update_plot()

                # begin the pre-process data timer
                self.pre_process_timer.start()

                # update plot timer
                self.timer2.start()
            else:
                print("Audio stream is not active")


    def pre_process_data(self):
        """data that is required by graphs but kept outside the update loop"""

        # we have to unify the source of the audio data,
        # because it handled in a separate thread and we need to synchronize it
        self.source = self.audio_handler.frame_buffer

        # make sure source is long enough
        if len(self.source) > 0:
            self.simplified_data = [] # our list of our simplified data
            self.negative_simplified_data = [] # our list of our simplified data

            # remove the unnecessary data only get the lowest and highest np
            for data in self.source:
                self.simplified_data.append(np.nanmax(data)) # [np.nanmin(data), np.nanmax(data)]
                self.negative_simplified_data.append(np.nanmin(data))

            # set the wave amplitude data
            self.wave_x = list(range(len(self.source)))
            self.wave_y = self.simplified_data
            self.wave_negative_y = self.negative_simplified_data
            

    amplitude_plot_a = None
    amplitude_plot_b = None
    fft_plot = None
    spectrogram_plot = None

    def update_plot(self, override = False):
        """Triggered by a timer to invoke canvas to update and redraw."""
        
        if override:
            print("override was invoked force update_plot!")

        if len(self.source) > 0:

            """
                Flush Events
            """
            # self.spg_canvas.flush_events()  # flush events
            # self.spg_canvas.axes.cla()      # Clear the canvas.

            """
                Amplitude Audio Plot
            """
            self.canvas.flush_events()  # flush events

            if self.amplitude_plot_a is None and self.amplitude_plot_b is None:
                # limit data range
                self.canvas.axes.set_ylim( [-2.5,2.5] )

                # label axis
                self.canvas.axes.set_xlabel( "Over Time", color="#ffffff" )

                # label axis
                self.canvas.axes.set_ylabel( "Amplitude", color="#ffffff" )
                self.canvas.axes.axhline(y=0, color='#353535', linestyle='-', alpha=1, label="0 Amplitude")

                self.amplitude_plot_a = self.canvas.axes.plot(self.wave_x, self.wave_y, '-', color="#7289da", alpha=1, label="Audio Signal")
                self.amplitude_plot_b = self.canvas.axes.plot(self.wave_x, self.wave_negative_y, '-', color="#7289da", alpha=1, label="Audio Signal")
            else:
                self.amplitude_plot_a[0].set_data(self.wave_x, self.wave_y)
                self.amplitude_plot_b[0].set_data(self.wave_x, self.wave_negative_y)

            self.canvas.draw_idle()     # actually draw the new content 
            
            """
                FFT Audio Plot
            """
            self.fft_canvas.flush_events()  # flush events
            #print(self.source[len(self.source)-1])

            try:
                # we need to numpy abs, because FFT will show negative frequencies and amplitudes
                self.fft_data = np.abs(np.fft.fft(self.source[len(self.source)-1]))
                self.fft_freq = np.abs(np.fft.fftfreq(len(self.fft_data), 1.0/config.SAMPLE_RATE))#np.fft.fftfreq(len(source[len(source)-1]), 1.0/config.SAMPLE_RATE)
            except Exception as e:
                print(f"Error when calculating fft data {e}")

            if self.fft_plot is None:
                 # define FFT plot limitations
                self.fft_canvas.axes.set_ylim(0, 150) # 300
                self.fft_canvas.axes.set_xlim(0, int(config.SAMPLE_RATE/2))

                # label axis
                self.fft_canvas.axes.set_xlabel( "Frequency", color="#ffffff" )

                # label axis
                self.fft_canvas.axes.set_ylabel( "Amplitude", color="#ffffff" )
                
                # plot the FFT
                self.fft_plot = self.fft_canvas.axes.plot(self.fft_freq, self.fft_data, '-', color="#eb4034", alpha=1, label="Audio Signal")
            else:
                #self.fft_plot[0].set_data(self.fft_freq, self.fft_data)
                self.fft_plot[0].set_ydata(self.fft_data)

            self.fft_canvas.draw_idle()     # actually draw the new content 
            """
                Spectrogram Audio Plot
            """
            # self.spg_canvas.flush_events()  # flush events

            # # rfft = np.fft.rfft(self.source[len(self.source)-1])

            # self.rfft_source.append(self.fft_data)

            # if len(self.rfft_source) > len(self.source):
            #     self.rfft_source.pop(0)
            
            # print(self.rfft_source)

            # #lb_fft = librosa.feature.melspectrogram(y=fft_data, sr=config.SAMPLE_RATE, S=None, n_fft=512, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0)
            
            # #D_highres = librosa.stft(fft_data, hop_length=256, n_fft=512) # 4096
            # #S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
            # #img = librosa.display.specshow(S_db_hr, hop_length=256, x_axis='time', y_axis='log',ax=self.spg_canvas.axes)
            
            # #S = librosa.feature.melspectrogram(y=self.wave_y, sr=config.SAMPLE_RATE)

            # #self.spg_canvas.axes.plot(fft_freq, fft_data, '-', color="#eb4034", alpha=1, label="Audio Signal")
            # #self.spg_canvas.axes.specgram(S, NFFT=512, Fs=config.SAMPLE_RATE, noverlap=256, cmap='jet' ) # noverlap=256, cmap='jet'
            
            # #self.spg_canvas.axes.specgram(self.fft_freq, NFFT=512, Fs=config.SAMPLE_RATE )#noverlap=256, cmap='jet' ) # noverlap=256, cmap='jet'



            # if self.spectrogram_plot is None:
                
            #     # label axes
            #     self.spg_canvas.axes.set_xlabel('Time', color="#ffffff")
            #     self.spg_canvas.axes.set_ylabel('Frequency', color="#ffffff")

            #     # def f(x):
            #     #     return np.int(x)

            #     # f2 = np.vectorize(f)

            #     #f2(self.source)

            #     # plot the FFT
            #     #self.spectrogram_plot = self.spg_canvas.axes.plot(self.rfft_source, '-', color="#eb4034", alpha=1, label="Audio Signal")#
            #     self.spg_canvas.axes.specgram(self.rfft_source, NFFT=512, Fs=config.SAMPLE_RATE ) #noverlap=256, cmap='jet' ) # noverlap=256, cmap='jet'
            #     #print(self.spectrogram_plot)
            # else:
            #     #self.spectrogram_plot[0].set_data(self.rfft_source)
            #     #pass
            #     self.spg_canvas.axes.cla()
            #     # #self.spectrogram_plot[0].set_data(list(range(len(self.source))), self.simplified_data)
            #     # # stft is short time fourier transform
            #     # X = librosa.stft(self.source)

            #     # # convert the slices to amplitude
            #     # Xdb = librosa.amplitude_to_db(abs(X))

            #     self.spg_canvas.axes.specgram(self.rfft_source, NFFT=512, Fs=config.SAMPLE_RATE ) #noverlap=256, cmap='jet' ) # noverlap=256, cmap='jet'

            #     # self.spg_canvas.axes.plot(self.fft_data, self.fft_freq, '-', color="#eb4034", alpha=1, label="Audio Signal")
            #     #self.spg_canvas.axes.specgram( (self.fft_freq,self.fft_data), NFFT=512, Fs=config.SAMPLE_RATE )
            #     # x = 0
            #     # for item in self.spectrogram_plot:
            #     #     print(x, item)
            #     #     x+= 1
            #     # #self.fft_plot[0].set_data(self.fft_freq, self.fft_data)
            #     # self.spectrogram_plot[1].set_data((self.fft_freq,self.fft_data))
            #     #pass

            # self.spg_canvas.draw_idle()     # actually draw the new content





        # if len(closes) > 0:

        #     # get bytes size of closes

        #     #print(plot_update_size,self.last_closes_size)
            
        #     # only update the chart if we have at least 2 elements to display
        #     if plot_update_size != self.last_closes_size or override:
                
        #         if not override:
        #             # only update plots if the last rsi is changed
        #             self.last_closes_size = plot_update_size

        #         """
        #             Trading view indicator canvas
        #         """
        #         self.canvas.flush_events()  # flush events
        #         self.canvas.axes.cla()      # Clear the canvas.
        #         self.canvas.draw_idle()     # actually draw the new content

        #         # candle close
        #         self.candle_xdata = list(range(len(closes)))
        #         self.candle_ydata = closes

        #         # # highs
        #         # self.candle_highs_xdata = list(range(len(highs)))
        #         # self.candle_highs_ydata = highs

        #         # # lows
        #         # self.candle_lows_xdata = list(range(len(lows)))
        #         # self.candle_lows_ydata = lows

        #         # sma MAX
        #         self.sma_max_values_xdata = list(range(len(sma_max)))
        #         self.sma_max_values_ydata = sma_max

        #         # sma half
        #         self.sma_half_values_xdata = list(range(len(sma_half)))
        #         self.sma_half_values_ydata = sma_half

        #         # sma diff
        #         self.sma_diff_values_xdata = list(range(len(sma_diff)))
        #         self.sma_diff_values_ydata = sma_diff

        #         # plot candle actual line
        #         self.canvas.axes.plot(self.candle_xdata, self.candle_ydata, '-', color="#7289da", alpha=0.4, label="Candle close price")

        #         # plot highs/lows of the candles
        #         # self.canvas.axes.plot(self.candle_highs_xdata, self.candle_highs_ydata, '-.', alpha=0.15, color="#2bff75", label="Candle close high price")
        #         # self.canvas.axes.plot(self.candle_lows_xdata, self.candle_lows_ydata, '-.', alpha=0.15, color="#fe2c54", label="Candle close low price")

        #         # plot SMA max and half and diff
        #         self.canvas.axes.plot(self.sma_max_values_xdata, self.sma_max_values_ydata, ':', color="#34ebeb", label="SMA full (" + str(RSI_PERIOD) + ")")
        #         self.canvas.axes.plot(self.sma_half_values_xdata, self.sma_half_values_ydata, ':', color="#2d55cc", label="SMA half (" + str(math.floor(RSI_PERIOD/2)) + ")")
                
        #         # make sure len closes is greater than RSI_PERIOD
        #         #if len(closes) > RSI_PERIOD:
        #         for i in range(0, len(sma_diff)):

        #             if not math.isnan(sma_diff[i]):
        #                 if sma_diff[i] > SMA_LONG: # 0.03
        #                     self.canvas.axes.plot(i, closes[i], 'x', color="#60ff60", label="SMA long" if i == 0 else "")
        #                 elif sma_diff[i] < SMA_SHORT: # -0.03
        #                     self.canvas.axes.plot(i, closes[i], 'x', color="#ff0000", label="SMA short" if i == 0 else "")
        #                 else:
        #                     self.canvas.axes.plot(i, closes[i], 'x', color="#909090", alpha=0.1, label="SMA Neutral" if i == 0 else "")
        #             #else:
        #                 #self.canvas.axes.plot(i, closes[i], 'x', color="#454545", label="SMA unknown" if i == 0 else "")

        #         # stop profit line
        #         if(last_stop_profit > 0):
        #             self.canvas.axes.axhline(y=last_stop_profit, color='#0d98ba', linestyle='-.', alpha=1, label="Stop profit")
                
        #         # stop profit limit line
        #         if(last_stop_profit_limit > 0):
        #             self.canvas.axes.axhline(y=last_stop_profit_limit, color='#0d98ba', linestyle='-', alpha=0.5, label="Stop profit limit")

        #         # if we are in position display where we bought in
        #         if in_position:           
        #             self.canvas.axes.axhline(y=last_buy_in_dict["avg"], color='#7d7dff', alpha=0.7, linestyle='--', label="Entry marker")
                
        #         # stop loss line
        #         if(last_stop_loss > 0):
        #             self.canvas.axes.axhline(y=last_stop_loss, color='#ff1515', linestyle='-.', alpha=1, label="Stop loss")

        #         # stop loss limit line
        #         if(last_stop_loss_limit > 0):
        #             self.canvas.axes.axhline(y=last_stop_loss_limit, color='#ff1515', linestyle='-', alpha=0.5, label="Stop loss limit")

        #         # plot trend line
        #         z = numpy.polyfit( self.candle_xdata, self.candle_ydata, 1)
        #         p = numpy.poly1d(z)
        #         self.canvas.axes.plot( self.candle_xdata, p( self.candle_xdata ),linestyle="--", color="#FF6C00", label="Trend line")

        #         # bollinger bands plots
        #         self.bb_upper_xdata = list(range(len(bb_upper)))
        #         self.bb_upper_ydata = bb_upper

        #         self.bb_middle_xdata = list(range(len(bb_middle)))
        #         self.bb_middle_ydata = bb_middle

        #         self.bb_lower_xdata = list(range(len(bb_lower)))
        #         self.bb_lower_ydata = bb_lower

        #         # plot bollinger bands
        #         self.canvas.axes.plot(self.bb_upper_xdata, self.bb_upper_ydata, '-', color="#ffc117", alpha=0.7)#, label="BB Upper")
        #         self.canvas.axes.plot(self.bb_middle_xdata, self.bb_middle_ydata, '-', color="#0dffa2", alpha=0.7)#, label="BB Middle")
        #         self.canvas.axes.plot(self.bb_lower_xdata, self.bb_lower_ydata, '-', color="#760dff", alpha=0.7)#, label="BB Lower")

        #         # legend for main canvas
        #         l_main = self.canvas.axes.legend(frameon=False, loc='lower left', fancybox=False, shadow=False)

        #         # replace text of the legend
        #         for text in l_main.get_texts():
        #             text.set_color("white")

        #         # background grid    
        #         self.canvas.axes.grid(color='#2c2f33', linestyle='--')

        #         """
        #             Volatility canvas
        #         """
        #         self.fft_canvas.flush_events()  # flush events
        #         self.fft_canvas.axes.cla()      # Clear the canvas.
        #         self.fft_canvas.draw_idle()     # actually draw the new content

        #         # atr
        #         self.atr_values_xdata = list(range(len(atr)))
        #         self.atr_values_ydata = atr

        #         ## volatility :: atr/ma
        #         #self.volatility_values_xdata = list(range(len(volatility)))
        #         #self.volatility_values_ydata = volatility
                
        #         # # atr_extremes
        #         # self.atr_e_values_xdata = list(range(len(atr_extremes)))
        #         # self.atr_e_values_ydata = atr_extremes

        #         # plot ATR
        #         if len(atr) > 0: #and len(volatility) > 0):
        #             self.fft_canvas.axes.plot(self.atr_values_xdata, self.atr_values_ydata, '-', color="#fffd80", label="ATR" )
        #             #self.fft_canvas.axes.plot(self.volatility_values_xdata, self.volatility_values_ydata, '--', color="#FF69B4", label="Volatility" )
        #             #self.fft_canvas.axes.plot(self.atr_e_values_xdata, self.atr_e_values_ydata, '-.', color="#FF8C00", label="ATR-E" )
                    
        #             # plot the line where that if below we are allowed to buy in
        #             self.fft_canvas.axes.axhline(y=ATR_VOLATILITY_DETECTION_THRESHOLD, color='#505050', linestyle='--', label="Volatility detection threshold")

        #             # atraverage = numpy.mean(atr)
        #             # print("atraverage",atraverage)
        #             # self.fft_canvas.axes.axhline(y=atraverage, color='#252590', linestyle='--', label="ATR Average")

        #             # legend for main canvas
        #             l2 = self.fft_canvas.axes.legend(frameon=False, loc='lower left', fancybox=False, shadow=False)

        #             # replace text of the legend
        #             for text in l2.get_texts():
        #                 text.set_color("white")

        #         # background grid    
        #         self.fft_canvas.axes.grid(color='#2c2f33', linestyle='--')

        #         """
        #             RSI canvas
        #         """
        #         self.spg_canvas.flush_events()  # flush events
        #         self.spg_canvas.axes.cla()      # Clear the canvas.
        #         self.spg_canvas.draw_idle()     # actually draw the new content

        #         # rsi indictor data
        #         self.rsi_values_xdata = list(range(len(rsi)))
        #         self.rsi_values_ydata = rsi

        #         # plot rsi current
        #         self.spg_canvas.axes.plot(self.rsi_values_xdata, self.rsi_values_ydata, '-', color="#DA70D6", label="RSI" )

        #         # plot rsi overbought/oversold setting
        #         self.spg_canvas.axes.axhline(y=RSI_OVERBOUGHT, color='#505050', linestyle='--', label="Overbought limit")
        #         self.spg_canvas.axes.axhline(y=RSI_OVERSOLD, color='#D8BFD8', linestyle='--', label="Oversold limit")

        #         # # Drop off the first y element, append a new one.
        #         # self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        #         # legend for rsi canvas
        #         l_rsi = self.spg_canvas.axes.legend(frameon=False, loc='lower left', fancybox=False, shadow=False)

        #         # replace text of the legend
        #         for text in l_rsi.get_texts():
        #             text.set_color("white")
                
        #         # background grid    
        #         self.spg_canvas.axes.grid(color='#2c2f33', linestyle='--')
                
        #         """
        #             MACD canvas
        #         """
        #         self.macd_canvas.flush_events()  # flush events
        #         self.macd_canvas.axes.cla()      # Clear the canvas.
        #         self.macd_canvas.draw_idle()     # actually draw the new content

        #         # MACD
        #         self.macd_xdata = list(range(len(macd)))
        #         self.macd_ydata = macd

        #         # SIGNAL
        #         self.macdsignal_xdata = list(range(len(macdsignal)))
        #         self.macdsignal_ydata = macdsignal

        #         # HIST
        #         self.macdhist_xdata = list(range(len(macdhist)))
        #         self.macdhist_ydata = macdhist

        #         # plot MACD and singal
        #         self.macd_canvas.axes.plot(self.macd_xdata, self.macd_ydata, '-', color="#7d7dff", label="MACD" )
        #         self.macd_canvas.axes.plot(self.macdsignal_xdata, self.macdsignal_ydata, '-', color="#ff7d7d", label="Signal" )

        #         # plot macd histogram
        #         colormat = numpy.where(self.macdhist_ydata>0, '#90ff90','#ff4545') # map numpay array of colours for matplot lib if values above 0 should be green and below 0 red

        #         self.macd_canvas.axes.bar(self.macdhist_xdata, self.macdhist_ydata, 0.5, color=colormat)  

        #         # legend for macd canvas
        #         l_macd = self.macd_canvas.axes.legend(frameon=False, loc='lower left', fancybox=False, shadow=False)

        #         # replace text of the legend
        #         for text in l_macd.get_texts():
        #             text.set_color("white")

        #         # background grid    
        #         self.macd_canvas.axes.grid(color='#2c2f33', linestyle='--')
                
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

            # update indicator text
            self.indicator_text_label.setText( self.indicator_text() )

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