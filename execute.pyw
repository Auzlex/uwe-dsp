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
        
        # image for spectrogram
        self.img = pg.ImageItem()
        # add the img to the spg canvas
        self.spg_canvas.addItem(self.img)

        # set the image array to zeros
        self.img_array = np.zeros((1000, int(config.CHUNK_SIZE/2+1)))

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        # setup the correct scaling for y-axis
        freq = np.arange(0,int(config.SAMPLE_RATE/2))
        #freq = np.arange((config.CHUNK_SIZE/2)+1)/(float(config.CHUNK_SIZE)/config.SAMPLE_RATE)
        
        # set the y-axis to the correct frequency scale
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])

        # set img scale
        self.img.scale((1./config.SAMPLE_RATE)*config.CHUNK_SIZE, yscale)

        # set the label of the canvas to show frequency on the Y axis
        self.spg_canvas.setLabel('left', 'Frequency', units='Hz')

        # prepare window for later use
        self.win = np.hanning(config.CHUNK_SIZE)

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

        #Overall_Layout.setRowStretch(1, 1)
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

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            #if np.array_equal(data_x,data_y):
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'amplitude':
                self.traces[name] = self.amplitude_canvas.plot(pen='c', width=3)
                self.amplitude_canvas.setYRange(-2.5, 2.5, padding=0)
                self.amplitude_canvas.setXRange(0, len(self.source), padding=0.005)

                self.amplitude_canvas.setMouseEnabled(x=False,y=False)  
                self.amplitude_canvas.enableAutoRange(axis='y', enable=True)
                self.amplitude_canvas.setAutoVisible(y=1.0)  
                self.amplitude_canvas.setAspectLocked(lock=False)  


            elif name == 'amplitude2':
                self.traces[name] = self.amplitude_canvas.plot(pen='c', width=3)
                self.amplitude_canvas.setYRange(-2.5, 2.5, padding=0)
                self.amplitude_canvas.setXRange(0, len(self.source), padding=0.005)
            
                self.amplitude_canvas.setMouseEnabled(x=False,y=False)  
                self.amplitude_canvas.enableAutoRange(axis='y', enable=True)
                self.amplitude_canvas.setAutoVisible(y=1.0)  
                self.amplitude_canvas.setAspectLocked(lock=False)  

            elif name == 'fft':
                self.traces[name] = self.fft_canvas.plot(pen=pg.mkPen({'color': "#ff2a00"}), width=3)
                self.fft_canvas.setYRange(0, 350, padding=0)
                self.fft_canvas.setXRange(0, int(self.audio_handler.SAMPLE_RATE/2), padding=0.005)
                
                self.fft_canvas.setMouseEnabled(x=False,y=False)  
                self.fft_canvas.enableAutoRange(axis='y', enable=True)
                self.fft_canvas.setAutoVisible(y=1.0)  
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
            self.wave_x = list(range(len(self.source)))
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
                self.img_array = np.roll(self.img_array, -1, 0) # roll down one row
                self.img_array[-1:] = psd[0:self.img_array.shape[1]] # only take the first half of the spectrum
                self.img.setImage(self.img_array, autoLevels=False) # set the image data

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