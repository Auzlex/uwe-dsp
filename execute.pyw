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
        self.spg_canvas = MplCanvas(self, titleinfo="spectrogram".upper(), width=5, height=4, dpi=70)
        
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
        #sys.stdout = MyStream()
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

    line1 = None
    line2 = None

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
            #powerSpectrum, frequenciesFound, time, imageAxis
            self.spg_canvas.flush_events()  # flush events

            if self.spectrogram_plot is None:

                # label axes
                self.spg_canvas.axes.set_xlabel('Time', color="#ffffff")
                self.spg_canvas.axes.set_ylabel('Frequency', color="#ffffff")


                #root_dir = os.path.dirname(os.path.realpath(__file__))
                #awr = audio.AudioWavReader(os.path.join(root_dir, "73733292392.wav"))
                self.spg_canvas.axes.set_ylim(-1, 1)
                #self.spectrogram_plot = self.spg_canvas.axes.plot(list(range(len(self.audio_handler.np_buffer))),self.audio_handler.np_buffer, '-', color="#eb4034", alpha=1, label="Audio Signal")
                

                self.spectrogram_plot = []
                #self.spectrogram_plot = self.spg_canvas.axes.plot(list(range(len(self.audio_handler.np_buffer))),self.audio_handler.np_buffer, '-', color="#eb4034", alpha=1, label="Audio Signal")
                spectrum, freqs, time, image = self.spg_canvas.axes.specgram(self.audio_handler.np_buffer, Fs=config.SAMPLE_RATE, NFFT=512 ) # NFFT=512 noverlap=256, cmap='jet' ) # noverlap=256, cmap='jet'
                
                #self.spectrogram_plot = self.spg_canvas.axes.plot(self.rfb_x, self.rfb_y, '-', color="#eb4034", alpha=1, label="Audio Signal")
                #self.spectrogram_plot = self.spg_canvas.axes.plot(awr.y, '-', color="#eb4034", alpha=1, label="Audio Signal")

                #self.spg_canvas.axes.specgram(awr.y, Fs=config.SAMPLE_RATE, NFFT=512 ) # NFFT=512 noverlap=256, cmap='jet' ) # noverlap=256, cmap='jet'
                

                #self.x = list(range(len(self.source)))
                #self.y = self.source # use the amplitude of the audio data as the y data
                #amplitude = np.fromstring(self.audio_handler.frame_buffer, np.int32)
                # create a line object with random data
                # variable for plotting
                #x = np.arange(0, 2 * self.audio_handler.CHUNK, 2) # 2 * self.audio_handler.CHUNK
                #x = list(range(len(self.audio_handler.raw_frame_buffer)))
                #x = np.linspace(0, SAMPLESIZE-1, SAMPLESIZE)
                # self.combined = []
                # for elem in self.audio_handler.raw_frame_buffer:
                #     print(elem)
                #     for num in elem:
                #         print(num)
                #         self.combined.append(elem)
                #self.spectrogram_plot, = self.spg_canvas.axes.plot(x,  self.audio_handler.raw_frame_buffer, '-', lw=2) # np.random.rand(self.audio_handler.CHUNK)

                #self.spectrogram_plot = self.spg_canvas.axes.plot(self.audio_handler.frame_buffer2, '-', color="#eb4034", alpha=1, label="Audio Signal")#

                # def f(x):
                #     return np.int(x)

                # f2 = np.vectorize(f)

                #f2(self.source)


                # self.line1 = self.spg_canvas.axes.plot([],[])[0]
                # self.line2 = self.spg_canvas.axes.plot([],[])[0]

                # self.line1.set_data(r, [-1000]*l)
                # self.line2.set_data(r, [-1000]*l)





                # plot the FFT
                #self.spectrogram_plot = self.spg_canvas.axes.plot(self.rfft_source, '-', color="#eb4034", alpha=1, label="Audio Signal")#
                ##self.spg_canvas.axes.specgram(self.source2, Fs=config.SAMPLE_RATE, NFFT=512 ) # NFFT=512 noverlap=256, cmap='jet' ) # noverlap=256, cmap='jet'
                #print(self.spectrogram_plot)
            else:
                self.spg_canvas.axes.cla()
                self.spg_canvas.axes.set_ylabel('Frequency', color="#ffffff")
                spectrum, freqs, time, image = self.spg_canvas.axes.specgram(self.audio_handler.np_buffer, Fs=config.SAMPLE_RATE, NFFT=512 ) # NFFT=512 noverlap=256, cmap='jet' ) # noverlap=256, cmap='jet'
                
                #self.spg_canvas.axes.set_ylim(-1, 1)
                #self.spg_canvas.axes.set_xlim(0, len(self.audio_handler.np_buffer))
                ##########self.spectrogram_plot[0].set_data(list(range(len(self.audio_handler.np_buffer))),self.audio_handler.np_buffer)
                #self.spectrogram_plot[0].set_data(list(range(len(self.audio_handler.np_buffer))),self.audio_handler.np_buffer)
                #self.spectrogram_plot[0].set_data(self.rfb_x, self.rfb_y)
                ##self.spectrogram_plot[0].set_data(self.rfb_x2, self.source2)

            self.spg_canvas.draw_idle()     # actually draw the new content

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