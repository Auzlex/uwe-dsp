#!/usr/bin/env python3
"""
    Import Modules
"""
import config # custom module that contains all the configuration information
import utility # custom module that contains all the utility functions
import tf_interface # custom module that contains tensorflow interface functions
import audio_handler as audio # custom audio handler class, handles audio input and output
import librosa # audio library
import os # used to determine if the app is running on windows or not
import sys # used tp tie in stdout and stderr to the console
import ctypes # used to access windows ctypes
import traceback # used for debugging
import threading
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# import for debugging
import logging

# import pyqt5 modules
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer#QSize, QTimer, 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QTextCursor, QPalette
from pyqtgraph.colormap import ColorMap
from pyqtgraph import Vector

#from matplotlib import cm # was used to get other colour maps

# configure the logging
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO)

# enable logging to verbose
logging.getLogger().setLevel(logging.DEBUG)

# disable numpy divide by zero warning
np.seterr(divide = 'ignore')

# application version
__version__ = '0.0.1'

"""
    PyQT5 Application
"""
global PYTHON_OUTPUT # stores our python output into console
PYTHON_OUTPUT = []

class StandardStream(object):
    """
        This function captures print statements
    """

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

class ApplicationWindow(QMainWindow):

    def __init__(self): # initialization function
        super().__init__() # invoke derived class constructor
        self.ltl = 0 # also used to gate keep update timer from spamming wasting cpu usage

        self.ai_keyed_in = False # determines if the AI is keyed in

        self.setup_user_interface() # invoke the ui initialize

    def key_in_augmented_intelligence(self):
        """
            Function: key_in_augmented_intelligence()
            Description: invoked when the user presses the key in button, this function will
            start the key in process which allows initializes the AI to listen into the audio data
        """
        logging.info("key in button pressed, starting key in process")
        print("listening...")
        self.ai_keyed_in = True

    def key_out_augmented_intelligence(self):
        """
            Function: key_out_augmented_intelligence()
            Description: invoked when the user presses the key out button, this function will
            stop the AI from listening into the audio data.
        """
        logging.info("key out button pressed, stopping key in process")
        print("stopped listening...")
        self.ai_keyed_in = False

    def indicator_text(self):
        """
            Function: indicator_text()
            Description: this function will update the indicator text
        """
        s = None

        # if the audio handler is not none
        if self.audio_handler is None:
            # ourput no audio handler
            s = "<font color=\'red\'>NO AUDIO HANDLER</font>"
        else:
            # get audio handler information
            if self.audio_handler.is_stream_active():
                s = (f"sample rate: <font color=\'orange\'>{self.audio_handler.stream._rate}</font> Hz | channels: <font color=\'cyan\'>{self.audio_handler.stream._channels}</font> | chunk: <font color=\'lime\'>{self.audio_handler.stream._frames_per_buffer}</font> | device_index: <font color=\'pink\'>{self.audio_handler.available_devices_information[self.mscb.currentIndex()]['index']}</font>").upper()
            else:
                s = (f"<font color=\'orange\'>AWAITING AUDIO STREAM INITIALIZATION</font>").upper()

        # return string
        return s

    def populate_microphone_sample_rates(self):

        index = self.mscb.currentIndex()
        selected_device_id = self.audio_handler.available_devices_information[index]['index']

        supported_srs = self.audio_handler.fetch_supported_sample_rates(selected_device_id)
        best_sr = supported_srs.index(int(self.audio_handler.available_devices_information[index]['defaultSampleRate']))
        
        self.sscb.setItems( [ str(x) for x in supported_srs ] )
        self.sscb.setCurrentIndex(best_sr)

    def microphone_selection_changed(self, value):
        """
            Function: microphone_selection_changed
            Description: invoked when the user changes the microphone selection
        """
        print(f"microphone DEVICE change requested -> switching to device: {self.audio_handler.available_devices_information[value]['name']}")

        if self.audio_handler is not None:

            # disconnect the sample rate change listener to prevent audio handler and devices spamming new stream creations
            try:
                self.sscb.currentIndexChanged.disconnect(self.samplerate_selection_changed)
            except Exception as e:
                print(f"disconnect error: {e}")

            self.populate_microphone_sample_rates()

            selected_device_id = self.audio_handler.available_devices_information[value]['index']
            supported_srs = self.audio_handler.fetch_supported_sample_rates(selected_device_id)
            best_sr = supported_srs.index(int(self.audio_handler.available_devices_information[value]['defaultSampleRate']))
            
            # #print(f"{supported_srs}, b: {best_sr}")
            # self.sscb.setItems( [ str(x) for x in supported_srs ] )
            # self.sscb.setCurrentIndex(best_sr)

            # reconnect the sample rate change listener so if a user wants to change they can do
            self.sscb.currentIndexChanged.connect(self.samplerate_selection_changed)

            # adjust the audio handler stream to new device
            self.audio_handler.adjust_stream(selected_device_id, supported_srs[best_sr])

            # if self.spectrogram_img_array is not None and self.spg_canvas is not None:
            #     self.adjust_spectrogram_scale(self.spectrogram_img, self.spectrogram_img_array, sr=supported_srs[best_sr])

        else:
            self.sscb.setItems( [ "No Audio Handler" ] )

    def samplerate_selection_changed(self, value):
        
        if self.audio_handler is not None:
            index = self.mscb.currentIndex()
            selected_device_id = self.audio_handler.available_devices_information[index]['index']
            supported_srs = self.audio_handler.fetch_supported_sample_rates(selected_device_id)
            print(f"SAMPLERATE change requested -> switching to sr: {supported_srs[value]} on device named: {self.audio_handler.available_devices_information[index]['name']}")
            
            # make sure the device id is 
            self.audio_handler.adjust_stream(selected_device_id, supported_srs[value])

    def fetch_h5_model(self):
        print("starting -> QFileDialog")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Hierarchical Data Format (*.H5)", options=options) #All Files (*);;
        
        # if the user selected a file
        if fileName:

            print(f"Hierarchical data format file provided: {fileName}")
        
            # disable the current AI
            self.key_out_augmented_intelligence()

            # attempt to initialize an instance of the tf_interface
            try:

                # initialize the tf_interface
                self.tf_model_interface = tf_interface.TFInterface(fileName)

            except Exception as e:
                print(f"Error when initializing the TFInterface {e}")

            # initialize the 3d visualizer
            if len(self.tf_model_interface.layers) > 0:
                self.generate_3d_visualization_of_tf_model(self.tf_model_interface.layers)

    def generate_3d_visualization_of_tf_model(self, layers:list):

        print("visualizing 3d model")

        self.glvw.clear()

        # z = pg.gaussianFilter(np.random.normal(size=(50,50)), (1,1))
        # p13d = gl.GLSurfacePlotItem(z=z, shader='shaded', color=(0.5, 0.5, 1, 1))
        # self.glvw.addItem(p13d)

        # md = gl.MeshData.sphere(rows=10, cols=20,radius=0.15)
        # m1 = gl.GLMeshItem(
        #     meshdata=md,
        #     smooth=True,
        #     color=(1, 1, 1, 1),
        #     shader="shaded",#"balloon",
        #     #glOptions="additive",
        # )
        # m1.resetTransform()
        # # x <right+, -left> z<+forward,-backward> y<+up, -down>
        # m1.translate(0, 0, 0)
        # self.glvw.addItem(m1)

        # md = gl.MeshData.sphere(rows=10, cols=20,radius=0.15)
        # m1 = gl.GLMeshItem(
        #     meshdata=md,
        #     smooth=True,
        #     color=(0, 0, 1, 1),
        #     shader="shaded",#"balloon",
        #     #glOptions="additive",
        # )
        # m1.resetTransform()
        # # x <right+, -left> z<+forward,-backward> y<+up, -down>
        # m1.translate(0, 1, 0)
        # self.glvw.addItem(m1)

        # md = gl.MeshData.sphere(rows=10, cols=20,radius=0.15)
        # m1 = gl.GLMeshItem(
        #     meshdata=md,
        #     smooth=True,
        #     color=(0, 1, 0, 1),
        #     shader="shaded",#"balloon",
        #     #glOptions="additive",
        # )
        # m1.resetTransform()
        # # x <right+, -left> z<+forward,-backward> y<+up, -down>
        # m1.translate(1, 0, 0)
        # self.glvw.addItem(m1)

        # md = gl.MeshData.sphere(rows=10, cols=20,radius=0.15)
        # m1 = gl.GLMeshItem(
        #     meshdata=md,
        #     smooth=True,
        #     color=(1, 1, 0, 1),
        #     shader="shaded",#"balloon",
        #     #glOptions="additive",
        # )
        # m1.resetTransform()
        # # x <right+, -left> z<+forward,-backward> y<+up, -down>
        # m1.translate(0, 0, 1)
        # self.glvw.addItem(m1)

        model_shape_parent = [
            
        ]

        a = 1
        N = 1
        colours = [
            (0, 0, N, 0.01),
            (N, N, 0, 0.01),
            (0, N, N, 0.01),
            (N, 0, N, 0.01)
        ]

        count = 0
        
        total_nn_shape = 0
        for i, layer in enumerate(layers):
            
            shape = layer[0]

            try:
                n3 = shape[2]
            except IndexError:
                n3 = 1

            total_nn_shape = total_nn_shape + n3

        next_shape_offset = 0
        for i, layer in enumerate(layers):
            count += 1

            if count > len(colours)-1:
                count = 0

            shape = layer[0]
            name = layer[1]

            n1 = shape[0]

            try:
                n2 = shape[1]
            except IndexError:
                n2 = 1
            
            try:
                n3 = shape[2]
            except IndexError:
                n3 = 1

            #print(layer,shape,name)
            model_shape_parent = []
            for x in range(n1):
                for y in range(n2):
                    for z in range(n3):
                        # X is 
                        model_shape_parent.append( [ (x - n1/2) * 0.05, 0 + (((next_shape_offset + ( i * 5 )) + z) - (total_nn_shape/2)) * 0.025, (y - (n2/2)) * 0.05 ] ) # [ x, z, i * 10 ]
                        #model_shape_parent.append( [ x * 5, (i * 150 * n3) + z, y * 5 ] ) # [ x, z, i * 10 ]
                        #model_shape_parent.append( [ x, z - int(z/2), y ] ) # [ x, z, i * 10 ]

            # get the last Z dim shape and append to next shape offset to ensure that the shapes are not overlapping
            next_shape_offset += n3

            # x <right+, -left> z<+forward,-backward> y<+up, -down>

            colour = (0.1, 0.1, 0.1, 0.025)#colours[count]#(0.1, 0.1, 0.1, 0.025) # all nodes are grey/white
            if i == 0:
                colour = (N, 0, 0, a) # entry nodes are green
            elif i >= len(layers)-2:
                colour = (0, N, 0, a) # label nodes are red

            m = gl.GLScatterPlotItem(pos=np.array(model_shape_parent), color=colour, size=5, pxMode=True)
            self.glvw.addItem(m)
            
        #self.grid = gl.GLGridItem(size=QtGui.QVector3D(int(next_shape_offset/2) * 0.025,(next_shape_offset * 0.025) * 2,1))
        self.grid = gl.GLGridItem()
        #self.grid.setTickSpacing(x=10, y=10, z=10)
        self.glvw.addItem(self.grid)

        self.glvw.setCameraPosition(pos=Vector( 0, ((next_shape_offset + len(layers) * 5) / 2 - (total_nn_shape/2)) * 0.025, 0 ), distance=(next_shape_offset / 2) * 0.025)
        del model_shape_parent

    def setup_user_interface(self) -> None:
        """
            Function: setup_user_interface()
            Description: this function will setup the UI
        """
        #self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        # add the key_in_augmented_intelligence function to the key_in_button
        key_in_button = QAction('Attach AI', self)
        key_in_button.setShortcut('Ctrl+Q')
        key_in_button.setToolTip('This button will engage the AI and tie it into the audio stream data.')
        key_in_button.triggered.connect(self.key_in_augmented_intelligence)

        # add the key_out_augmented_intelligence function to the key_in_button
        key_out_button = QAction('Detach AI', self)
        key_out_button.setShortcut('Ctrl+W')
        key_out_button.setToolTip('This button will disengage the AI from the audio stream data.')
        key_out_button.triggered.connect(self.key_out_augmented_intelligence) 

        # add the key_out_augmented_intelligence function to the key_in_button
        load_ml_model_button = QAction('Load H5 Model', self)
        #load_ml_model_button.setShortcut('Ctrl+L')
        load_ml_model_button.setToolTip('This button will invoke a file dialog and allows you to load a TF h5 model.')
        load_ml_model_button.triggered.connect(self.fetch_h5_model) 

        """Microphone Selection ComboBox"""
        # label
        self.mscb_label = QLabel()
        self.mscb_label.setText( "Target Microphone:" )

        self.mscb = pg.ComboBox()

        """Samplerate Selection ComboBox"""
        # label
        self.sscb_label = QLabel()
        self.sscb_label.setText( "Target Samplerate:" )

        self.sscb = pg.ComboBox()
        #items = {'a': 1, 'b': 2, 'c': 3}
        #self.sscb.setItems(items)
        #self.sscb.setValue(1)

        """Master Control Bar"""
        # add the buttons to the toolbar
        self.toolbar = self.addToolBar('Master Control Toolbar')
        self.toolbar.addAction(key_in_button)
        self.toolbar.addAction(key_out_button)
        self.toolbar.addWidget(self.mscb_label)
        self.toolbar.addWidget(self.mscb)
        self.toolbar.addWidget(self.sscb_label)
        self.toolbar.addWidget(self.sscb)
        self.toolbar.addAction(load_ml_model_button)

        """Initial Variables """
        self.tf_interface = None # type: tf_interface.TFInterface
        self.audio_handler = None # set to none so that we can check if it is None later
        self.source = [] # source of our frame buffer

        self.fft_data = [] # data for fft
        self.fft_freq = [] # frequency for fft

        # get the file directory
        root_dir_path = utility.get_root_dir_path()

        """label indicator"""
        # label indicator
        self.indicator_text_label = QLabel()

        # update indicator text
        self.indicator_text_label.setText( self.indicator_text() )

        # console output
        self.textEdit = QTextEdit()
        self.textEdit.setReadOnly(True)
        self.textEdit.verticalScrollBar().setValue(1)

        """Pyqtplot render target for audio"""
        pg.setConfigOptions(antialias=True) # set antialiasing on for prettier plots
        self.audio_pyqtplot_rendertarget = pg.GraphicsLayoutWidget(title="graphics window".upper())

        # has a list of all our plots that we will dynamically update
        self.traces = dict()

        """ Canvas References for Plotting"""
        self.amplitude_canvas = self.audio_pyqtplot_rendertarget.addPlot(title="audio amplitude".upper(), row=0, col=0)
        self.fft_canvas = self.audio_pyqtplot_rendertarget.addPlot(title="Fourier Wave Transform".upper(), row=1, col=0)
        self.spg_canvas = self.audio_pyqtplot_rendertarget.addPlot(title="Linear-scale spectrogram".upper(), row=2, col=0)
        self.mel_spec_canvas = self.audio_pyqtplot_rendertarget.addPlot(title="Mel-scale spectrogram".upper(), row=3, col=0)
        self.mfcc_spec_canvas = self.audio_pyqtplot_rendertarget.addPlot(title="Mel-frequency cepstral coefficients".upper(), row=4, col=0)

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
        
        # set the y-axis to the correct frequency scale
        yscale = 1.0/(self.spectrogram_img_array.shape[1]/freq[-1])

        # set spectrogram_img scale
        self.spectrogram_img.scale((1./config.SAMPLE_RATE)*config.CHUNK_SIZE, yscale)

        # set the label of the canvas to show frequency on the Y axis
        self.spg_canvas.setLabel('left', 'Frequency', units='Hz')
        #self.spg_canvas.setXRange(0, config.MAX_BUFFER_TIME * 2, padding=0) # this will limit the spectrogram to the last 10 seconds of data and stop it spazzing

        """Mel Spectrogram plot properties and setup"""
        # image for mel spectrogram
        self.mel_spectrogram_img = pg.ImageItem()
        # add the mel_spectrogram_img to the mel_spec canvas
        self.mel_spec_canvas.addItem(self.mel_spectrogram_img)
        # set the image array to zeros
        self.mel_spectrogram_img_array = np.zeros((1000, int(config.CHUNK_SIZE/2+1)))

        # myLUT = np.array([[1.        , 1.        , 1.        ],
        #           [0.38401946, 0.48864573, 0.963664  ],
        #           [0.28766167, 0.81375253, 0.49518645],
        #           [0.71970558, 0.92549998, 0.34362429]]) * 255

        # set colormap
        self.mel_spectrogram_img.setLookupTable(lut)
        self.mel_spectrogram_img.setLevels([-3,0.5])

        # setup the correct scaling for y-axis
        freq = np.arange(0,int(config.SAMPLE_RATE/2))
        
        # set the y-axis to the correct frequency scale
        yscale = 1.0/(self.mel_spectrogram_img_array.shape[1]/freq[-1])

        # set spectrogram_img scale
        self.mel_spectrogram_img.scale((1./config.SAMPLE_RATE)*config.CHUNK_SIZE, yscale)

        # set the label of the canvas to show frequency on the Y axis
        self.mel_spec_canvas.setLabel('left', 'Frequency', units='Hz')

        """MFCC Spectrogram plot properties and setup"""
        # image for mel spectrogram
        self.mfcc_spectrogram_img = pg.ImageItem()
        # add the mel_spectrogram_img to the mel_spec canvas
        self.mfcc_spec_canvas.addItem(self.mfcc_spectrogram_img)
        # set the image array to zeros
        self.mfcc_spectrogram_img_array = np.zeros((1000, int(config.CHUNK_SIZE/2+1)))

        # set colormap
        self.mfcc_spectrogram_img.setLookupTable(lut)
        self.mfcc_spectrogram_img.setLevels([-1,1]) # [-50,40]

        # setup the correct scaling for y-axis
        freq = np.arange(0,int(config.SAMPLE_RATE/2))
        
        # set the y-axis to the correct frequency scale
        yscale = 1.0/(self.mfcc_spectrogram_img_array.shape[1]/freq[-1])

        # set spectrogram_img scale
        self.mfcc_spectrogram_img.scale((1./config.SAMPLE_RATE)*config.CHUNK_SIZE, yscale)

        # set the label of the canvas to show frequency on the Y axis
        self.mfcc_spec_canvas.setLabel('left', 'Frequency', units='Hz')

        """construct the 3d viewer window with pyqtgraph"""
        ## test
        p1 = pg.PlotWidget()

        # try adding a 3d plot
        self.glvw = gl.GLViewWidget()

        self.grid = gl.GLGridItem()
        #self.grid.setTickSpacing(x=10, y=10, z=10)
        self.glvw.addItem(self.grid)

        # z = pg.gaussianFilter(np.random.normal(size=(50,50)), (1,1))
        # p13d = gl.GLSurfacePlotItem(z=z, shader='shaded', color=(0.5, 0.5, 1, 1))
        # self.glvw.addItem(p13d)

        md = gl.MeshData.sphere(rows=10, cols=20,radius=0.15)
        m1 = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(1, 1, 1, 1),
            shader="shaded",#"balloon",
            #glOptions="additive",
        )
        m1.resetTransform()
        # x <right+, -left> z<+forward,-backward> y<+up, -down>
        m1.translate(0, 0, 0)
        self.glvw.addItem(m1)

        md = gl.MeshData.sphere(rows=10, cols=20,radius=0.15)
        m1 = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(0, 0, 1, 1),
            shader="shaded",#"balloon",
            #glOptions="additive",
        )
        m1.resetTransform()
        # x <right+, -left> z<+forward,-backward> y<+up, -down>
        m1.translate(0, 1, 0)
        self.glvw.addItem(m1)

        md = gl.MeshData.sphere(rows=10, cols=20,radius=0.15)
        m1 = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(0, 1, 0, 1),
            shader="shaded",#"balloon",
            #glOptions="additive",
        )
        m1.resetTransform()
        # x <right+, -left> z<+forward,-backward> y<+up, -down>
        m1.translate(1, 0, 0)
        self.glvw.addItem(m1)

        md = gl.MeshData.sphere(rows=10, cols=20,radius=0.15)
        m1 = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(1, 1, 0, 1),
            shader="shaded",#"balloon",
            #glOptions="additive",
        )
        m1.resetTransform()
        # x <right+, -left> z<+forward,-backward> y<+up, -down>
        m1.translate(0, 0, 1)
        self.glvw.addItem(m1)

        # np.random.normal(size=(2500, 3))

        # create empty numpy array with 3 dims 
        # array = np.zeros((4, 3))
        # print(array)

        # md = gl.GLScatterPlotItem(pos=array, color=(1, 0, 0, 1))
        # self.glvw.addItem(md)

        # array = np.random.normal(size=(10, 3))

        # # create empty numpy array with 3 dims 
        # #array = np.zeros((4, 3))
        # print(array)

        # md = gl.GLScatterPlotItem(pos=array, color=(1, 0, 0, 1))
        # self.glvw.addItem(md)

        """
            https://stackoverflow.com/questions/52704327/how-to-insert-a-3d-glviewwidget-into-a-window-containing-2d-pyqtgraph-plots
            the PlotWidget has more aggressive default settings because it inherits from QGraphicsView" 
            - source I have yet to understand PyQT(Graph) and OpenGL, 
            so I'm sorry I can't say much more, but these 3 lines should solve your motivating example:
        """
        p1.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.glvw.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.glvw.setSizePolicy(p1.sizePolicy())

        """Setup, GUI layout"""
        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(500) # 500
        self.timer.timeout.connect(self.update_console) # self.update_plot

        # Setup a timer to trigger the redraw by calling update_plot.
        self.pre_process_timer = QTimer()
        self.pre_process_timer.setInterval(0) # 500
        self.pre_process_timer.timeout.connect(self.pre_process_data) # self.update_plot

        # Setup a timer to trigger the redraw by calling update_plot.
        self.ai_classify_timer = QTimer()
        self.ai_classify_timer.setInterval(2000) # 500
        self.ai_classify_timer.timeout.connect(self.classify_audio_update) # self.update_plot

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer2 = QTimer()
        self.timer2.setInterval(0) # 3000
        self.timer2.timeout.connect(self.update_plot) # self.update_plot

        # setup the layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # create the layout
        Overall_Layout = QGridLayout(central_widget)
        Overall_Layout.setRowStretch(3, 3) # 3, 3)
        Overall_Layout.addWidget( self.indicator_text_label, 1, 1 )
        Overall_Layout.addWidget( self.textEdit, 2, 1 )
        Overall_Layout.addWidget( self.audio_pyqtplot_rendertarget, 3, 1 )
        #Overall_Layout.setRowStretch(2, 3)
        #Overall_Layout.addWidget( self.glvw, 2, 2 ) # append the 3d plot to the layout
        Overall_Layout.addWidget( self.glvw, 3, 2 ) # append the 3d plot to the layout

        # setup the window layout
        self.setGeometry(300, 300, 1280, 720)        # set the size of the window
        self.setWindowTitle('audio signal analysis classification with neural networks'.upper())              # set window title Audio.To.SpectroGraph # multi-label sound event classification system
        self.setWindowIcon(QIcon(os.path.join( root_dir_path, 'icon.png' )))       # set window icon

        # display the window
        self.show()

        # hook events for python program out so that we can view debug information of the last 250 characters
        sys.stdout = StandardStream()
        #sys.stderr = StandardStream(True)

        # update text timer
        self.timer.start()

        # attempt to initialize the audio handler object and start the audio stream
        try:
            self.audio_handler = audio.AudioHandler()

            # populate the combox box with the available audio devices
            self.mscb.setItems( [ f"{item['name']}" for x,item in enumerate(self.audio_handler.available_devices_information)] )
 
            self.populate_microphone_sample_rates()

            # connect the functions to listen for changes in the audio device selection
            self.mscb.currentIndexChanged.connect(self.microphone_selection_changed)
            self.sscb.currentIndexChanged.connect(self.samplerate_selection_changed)

            # set the starting device to be the first found input device
            self.mscb.setCurrentIndex(0)
            index = self.mscb.currentIndex()
            selected_device_id = self.audio_handler.available_devices_information[index]['index']

            # this code starts the new audio stream of desired device and sample rate i.e the first one in the list
            self.audio_handler.adjust_stream(selected_device_id, int(self.audio_handler.available_devices_information[index]['defaultSampleRate']))
            
            # https://www.tutorialspoint.com/pyqt/pyqt_qfiledialog_widget.htm

            # start the audio stream
            #self.audio_handler.start()
        except Exception as e:
            print(f"Error when initializing and starting audio handler {e}")
            logging.critical("Error: %s" % e)
        else:
            if self.audio_handler is not None:
                if self.audio_handler.is_stream_active():
                    print("Audio stream is active")

                # begin the timers
                self.pre_process_timer.start() # start the pre-processing timer
                self.ai_classify_timer.start() # start the AI classification timer

                # update plots
                self.update_plot()

                # update plot timer
                self.timer2.start()
            else:
                print("Audio stream is not active")

        #TODO: REMOVE THIS AFTER DEBUGGING VISUAIZER
        # attempt to initialize an instance of the tf_interface
        #try:

        # initialize the tf_interface
        self.tf_model_interface = tf_interface.TFInterface("/home/charlesedwards/Documents/final_models/MFCC_CNN_lr-0.0001_b1-0.99_b2-0.999_EPOCH-500_BATCH-32_cc_v3.h5")
        print(self.tf_model_interface.metadata)

        # except Exception as e:
        #     print(f"Error when initializing the TFInterface {e}")

        # initialize the 3d visualizer
        if len(self.tf_model_interface.layers) > 0:
            self.generate_3d_visualization_of_tf_model(self.tf_model_interface.layers)

        #self.tf_model_interface = None

        # attempt to initialize an instance of the tf_interface
        #try:

        # # initialize the tf_interface
        # self.tf_model_interface = tf_interface.TFLiteInterface(
        #     os.path.join( 
        #         root_dir_path, 
        #         'tf_models', 
        #         'yamnet_classification_1.tflite'
        #     ),
        #     #24#self.audio_handler.np_buffer.size # size of our waveform
        # )

        
        # def _plot_dots(layers_array, layers_name, layers_color, layers_marker, ax, xy_max):
        #     """plot layers units as dots"""
        #     temp = True
        #     last_a, last_b, last_c = [0, 0], [0, 0], [0, 0]

        #     #scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(30, 255, 35, 255))

        #     for layer, name, color_in, marker in zip(layers_array, layers_name, layers_color, layers_marker):
        #         line_x, line_y, line_z = [], [], []
        #         color_count = 0

        #         for j in layer:
        #             my_x, my_y, my_z = [], [], []
        #             temp_list_l = []

        #             for k in j[0]:
        #                 k = [a + ((xy_max[0] - len(k)) / 2) for a in k]
        #                 my_x += k

        #             line_x.append([k[0], k[-1]])

        #             for l in j[1]:
        #                 l = [b + ((xy_max[1] - (j[1][-1][-1] + 1)) / 2) for b in l]
        #                 my_y += l
        #                 temp_list_l.append(l[0])

        #             line_y.append([temp_list_l[0], temp_list_l[-1]])

        #             for k in j[2]:
        #                 my_z += k

        #             line_z.append([k[0], k[-1]])

        #             if color_in == 'rgb':
        #                 color = color_in[color_count]
        #                 color_count += 1
        #             else:selected_device_id = self.audio_handler.available_devices_information[index]['index']
        #                 color = color_in

                    

        #             ax.scatter3d(my_x, my_z, my_y, c=color, marker=marker)
        #             # # adding spots to the scatter plot
        #             # scatter.addPoints(x_data, y_data)
            
        #             # # add item to plot window
        #             # # adding scatter plot item to the plot window
        #             # plot.addItem(scatter)

        # #_plot_dots(self.tf_model_interface.layers_array, self.tf_model_interface.layers_name, self.tf_model_interface.layers_color, self.tf_model_interface.layers_marker, self.ml_v_canvas.axes, self.tf_model_interface.xy_max)

        # #self.ml_v.plot( self.tf_model_interface. )

        #     # adjust the size of the tensor
        #     #self.tf_interface.resize_tensor_input(self.audio_handler.np_buffer.size)

        # # except Exception as e:
        # #     print(f"Error when initializing the TFLiteInterface {e}")

    def set_plotdata(self, name, data_x, data_y, auto_scale=True) -> None:
        """
            Function: set_plotdata
            Description: set the data for the plot depending on the name of the plot
        """

        # if the given name is in the self.traces dictionary
        if name in self.traces:
            # then update the data
            self.traces[name].setData(data_x, data_y)
        else:

            # if the name is not in the self.traces dictionary, then create a new plot
            if name == 'amplitude':
                self.traces[name] = self.amplitude_canvas.plot(pen=pg.mkPen({'color': "#3736FF"}), width=3)
                self.amplitude_canvas.setYRange(-2.5, 2.5, padding=0)
                self.amplitude_canvas.setXRange(0, len(self.source), padding=0.005)

                if auto_scale:
                    self.amplitude_canvas.setMouseEnabled(x=False,y=False)  
                    self.amplitude_canvas.enableAutoRange(axis='y', enable=True)
                    self.amplitude_canvas.setAutoVisible(y=1.0)  
                    self.amplitude_canvas.setAspectLocked(lock=False)  

            elif name == 'amplitude2':
                self.traces[name] = self.amplitude_canvas.plot(pen=pg.mkPen({'color': "#3736FF"}), width=3) # "c" pg.mkPen({'color': "#3736FF"})
                self.amplitude_canvas.setYRange(-2.5, 2.5, padding=0)
                self.amplitude_canvas.setXRange(0, len(self.source), padding=0.005)
            
                if auto_scale:
                    self.amplitude_canvas.setMouseEnabled(x=False,y=False)  
                    self.amplitude_canvas.enableAutoRange(axis='y', enable=True)
                    self.amplitude_canvas.setAutoVisible(y=1.0)  
                    self.amplitude_canvas.setAspectLocked(lock=False)  

            elif name == 'fft':
                self.traces[name] = self.fft_canvas.plot(pen=pg.mkPen({'color': "#FF5349"}), width=3) # #ff2a00
                self.fft_canvas.setYRange(0, 250, padding=0)
                self.fft_canvas.setXRange(0, int(self.audio_handler.SAMPLE_RATE/2), padding=0.005)
                
                if auto_scale:
                    self.fft_canvas.setMouseEnabled(x=False,y=False)  
                    self.fft_canvas.enableAutoRange(axis='y', enable=True)
                    self.fft_canvas.setAutoVisible(y=3.0)  
                    self.fft_canvas.setAspectLocked(lock=False)  

    def pre_process_data(self):
        """data that is required by graphs but kept outside the update loop"""

        # we have to unify the source of the audio data,
        # because it handled in a separate thread and we need to synchronize it
        self.source = self.audio_handler.frame_buffer

        if len(self.source) > 0:
            try:
                # we need to numpy abs, because FFT will show negative frequencies and amplitudes
                self.fft_data = np.abs(np.fft.fft(self.source[len(self.source)-1]))
                self.fft_freq = np.abs(np.fft.fftfreq(len(self.fft_data), 1.0/self.audio_handler.stream._rate))#np.fft.fftfreq(len(source[len(source)-1]), 1.0/self.audio_handler.stream._rate)
            except Exception as e:
                pass
                #print(f"Error when calculating fft data {e}")

        # make sure source is long enough
       
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

    # def mel_spectrogram(self):
    #     """
    #         Function: mel_spectrogram
    #         Description: calculate the mel spectrogram
    #     """

    #     # calculate the mel spectrogram
    #     self.mel_spectrogram_data = librosa.feature.melspectrogram(self.source[len(self.source)-1], sr=self.audio_handler.stream._rate, n_mels=128, fmax=8000)
    #     self.mel_spectrogram_data = librosa.power_to_db(self.mel_spectrogram_data)

    #     # set the mel spectrogram data
    #     self.mel_spectrogram_x = list(range(len(self.mel_spectrogram_data[0])))
    #     self.mel_spectrogram_y = self.mel_spectrogram_data

    # # function that takes in audio data and erturns mel spectrogram
    # def get_mel_spectrogram(self, audio_data):
    #     """
    #         Function: get_mel_spectrogram
    #         Description: calculate the mel spectrogram
    #     """

    #     # calculate the mel spectrogram
    #     mel_spectrogram_data = librosa.feature.melspectrogram(audio_data, sr=self.audio_handler.stream._rate, n_mels=128, fmax=8000)
    #     mel_spectrogram_data = librosa.power_to_db(mel_spectrogram_data)

    #     # set the mel spectrogram data
    #     mel_spectrogram_x = list(range(len(mel_spectrogram_data[0])))
    #     mel_spectrogram_y = mel_spectrogram_data

    #     return mel_spectrogram_x, mel_spectrogram_y

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
                #np_array = librosa.feature.melspectrogram(y=self.audio_handler.np_buffer, sr=self.audio_handler.stream._rate, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0)
                
                # # # normalized, windowed frequencies in data chunk
                # old_spec = np.fft.rfft(self.source[len(self.source)-1])#self.audio_handler.np_buffer#np.fft.rfft(config.CHUNK_SIZE*self.win) / config.CHUNK_SIZE

                # # get magnitude 
                # psd = abs(old_spec)
                
                # # convert to dB scale
                # psd_t = 20 * np.log10(psd)

                stft = librosa.stft(self.audio_handler.np_buffer, n_fft=config.CHUNK_SIZE, hop_length=None)
                
                #convert to db
                stft_db_abs = librosa.amplitude_to_db(np.abs(stft))
                stft_db_abs_t = np.transpose(stft_db_abs)
                #psd_t = psd_t[len(psd_t)-1]
                #print(psd_t.shape,  self.spectrogram_img_array.shape)
                
                self.spectrogram_img.setImage(stft_db_abs_t, autoLevels=False)

                #print(psd_t[0:self.spectrogram_img_array.shape[1]])

                # self.spectrogram_img_array = np.roll(self.spectrogram_img_array, -1, 0) # roll down one row
                # self.spectrogram_img_array[-1:] = psd_t[0:self.spectrogram_img_array.shape[1]] # only take the first half of the spectrum
                # self.spectrogram_img.setImage(self.spectrogram_img_array, autoLevels=False) # set the image data


                # roll down one and replace leading edge with new data
                # self.spectrogram_img_array = np.roll(self.spectrogram_img_array, -1, 0) # roll down one row
                # self.spectrogram_img_array[-1:] = psd_t[0:self.spectrogram_img_array.shape[1]] # only take the first half of the spectrum
                # self.spectrogram_img.setImage(self.spectrogram_img_array, autoLevels=False) # set the image data

                """mel spectrogram""" 

                msg = librosa.feature.melspectrogram(S=stft_db_abs, sr=self.audio_handler.stream._rate)
                msg_t = np.transpose(msg) # transpose the data because for some reason they are in a weird format idk
                #msg_t_n = librosa.util.normalize(msg_t)
                self.mel_spectrogram_img.setImage(msg_t, autoLevels=False)

                #print(np.min(msg_t), np.max(msg_t))

                """mfcc""" 
                self.mfcc_sg = librosa.feature.mfcc(S=stft_db_abs, sr=self.audio_handler.stream._rate)
                mfcc_sg_t = np.transpose(self.mfcc_sg) # transpose the data because for some reason they are in a weird format idk
                mfcc_t_n = librosa.util.normalize(mfcc_sg_t)
                self.mfcc_spectrogram_img.setImage(mfcc_t_n, autoLevels=False)

                #print(np.min(mfcc_sg_t), np.max(mfcc_sg_t))

                # #n_fft was 2048, hop len 512 n_mels 128
                # mel_spec = librosa.feature.melspectrogram(y=self.audio_handler.np_buffer, sr=self.audio_handler.stream._rate, n_fft=config.CHUNK_SIZE, S=None, n_mels=128)
                # mel_spec = librosa.amplitude_to_db(np.abs(mel_spec))
                # mel_spec = np.transpose(mel_spec)
                # mel_spec = mel_spec[len(mel_spec)-1]

                # # # convert to dB scale
                # # mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                # # #mel_spec = mel_spec.flatten()

                # # # 
                # # mel_spec = np.expand_dims(mel_spec, axis=0)
                # # print(mel_spec.shape, self.mel_spectrogram_img_array.shape)
                # # #mel_spec = 20 * np.log10(mel_spec)


                # # broadcast mel_spec into mel_spectrogram_img_array and flattern array
                # self.mel_spectrogram_img_array = np.roll(self.mel_spectrogram_img_array, -1, 0) # roll down one row
                # self.mel_spectrogram_img_array[-1:] = mel_spec[0:self.mel_spectrogram_img_array.shape[1]] # only take the first half of the spectrum
                # self.mel_spectrogram_img.setImage(self.mel_spectrogram_img_array, autoLevels=False) # set the image data





                # fetch mel spectrogram data
                #mel_spectrogram_x, mel_spectrogram_y = self.get_mel_spectrogram(self.source[len(self.source)-1])

                # use spec and calculate the mel spectrogram
                





                # update the mel spectrogram via 

                # #n_fft was 2048, hop len 512
                # mel_spec = librosa.feature.melspectrogram(y=self.audio_handler.np_buffer, sr=self.audio_handler.stream._rate, S=None, n_fft=2048, hop_length=200, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0)[-1]

                # # convert to dB scale
                # mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                # #mel_spec = mel_spec.flatten()

                # # 
                # mel_spec = np.expand_dims(mel_spec, axis=0)
                # print(mel_spec.shape, self.mel_spectrogram_img_array.shape)
                # #mel_spec = 20 * np.log10(mel_spec)


                # # broadcast mel_spec into mel_spectrogram_img_array and flattern array
                # self.mel_spectrogram_img_array = np.roll(self.mel_spectrogram_img_array, -1, 0) # roll down one row
                # self.mel_spectrogram_img_array[-1:] = mel_spec[0:self.mel_spectrogram_img_array.shape[1]] # only take the first half of the spectrum
                # self.mel_spectrogram_img.setImage(self.mel_spectrogram_img_array, autoLevels=False) # set the image data


                # # roll down one and replace leading edge with new data
                # self.mel_spectrogram_img_array = np.roll(self.mel_spectrogram_img_array, -1, 0) # roll down one row
                # #self.mel_spectrogram_img_array[-1:] = mel_spec#mel_spec[0:self.mel_spectrogram_img_array.shape[1]] # only take the first half of the spectrum
                # self.mel_spectrogram_img.setImage(self.mel_spectrogram_img_array, autoLevels=False) # set the image data

    def update_console(self):
        """Triggered by a timer to invoke an update on text edit"""

        global PYTHON_OUTPUT

        # only update on change
        if len(PYTHON_OUTPUT) != self.ltl:

            # update last count
            self.ltl = len(PYTHON_OUTPUT)

            self.textEdit.clear()

            full_string = ""
            for line in PYTHON_OUTPUT:
                full_string += line

            self.textEdit.insertPlainText( str(full_string) )
            self.textEdit.moveCursor(QTextCursor.End)

            # # update indicator text
            # self.indicator_text_label.setText( self.indicator_text() )
 
    def perform_tf_classification(self):
        """perform a classification using the tf_interface"""
        
        #print("performing tf prediction")
        if self.mfcc_sg is not None and self.ai_keyed_in and self.tf_model_interface.model is not None:
            print(self.tf_model_interface.predict_mfcc( librosa.util.normalize(librosa.feature.mfcc(y=self.audio_handler.np_buffer, sr=self.audio_handler.stream._rate) ) ))
        else:
            print("no mfcc or no model")
        #pass
        # try:
        #     if self.tf_model_interface is not None:
        #         # convert the self.audio_handler.frame_buffer from its sample rate to 16khz
        #         new_buffer = self.audio_handler.resample(self.audio_handler.np_buffer, self.audio_handler.SAMPLE_RATE, 16000).astype(np.float32)
        #         #print(type(new_buffer), new_buffer.shape)
        #         #new_buffer = np.cast(new_buffer, dtype=np.float32) # convert to float32

        #         # shove in our wave form into the TF Lite Model
        #         #self.tf_interface.resize_tensor_input(new_buffer.size)
        #         self.tf_model_interface.feed(new_buffer)
        #         print(self.tf_model_interface.labels[self.tf_model_interface.fetch_best_score_index()])
        # except Exception:
        #     print(f"Error performing TF prediction {traceback.format_exc()}")

        # # make sure we have data
        # if len(self.source) > 0:
        #     # get the data
        #     data = self.source[len(self.source)-1]

        #     # get the classification
        #     classification = self.tf_model_interface.get_classification(data)

        #     # update the label
        #     self.label.setText(f"{classification}")

    def classify_audio_update(self):
        # make sure source is long enough
        if len(self.source) > 0:
            
            tf_thread = threading.Thread(target=self.perform_tf_classification)
            tf_thread.start()
        
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

    # detect if the app is running on the NT kernel if so we are on windows baby!
    if os.name.lower() == "nt":
        myappid = u'pyqt5.mlsecs.' + str(__version__) # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid) # adjust the app id of windows to the correct app id

    app = QApplication(sys.argv) # run the q application by passing in any system arguments through from the script
    
    # this section of code changes the app style to fusion i.e. dark theme and overrides the pallet to dark colours
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QtGui.QColor(0,0,0)) # 53
    palette.setColor(QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QPalette.Base, QtGui.QColor(0,0,0)) # 15
    palette.setColor(QPalette.AlternateBase, QtGui.QColor(20,20,20)) # 53
    palette.setColor(QPalette.ToolTipBase, QtGui.QColor(20,20,20))
    palette.setColor(QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QPalette.Text, QtCore.Qt.white)
    palette.setColor(QPalette.Button, QtGui.QColor(20,20,20)) # 53
    palette.setColor(QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QPalette.BrightText, QtCore.Qt.red) 
    palette.setColor(QPalette.Highlight, QtGui.QColor(103,119,204).lighter()) # 142,45,197
    palette.setColor(QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    # create the main window
    baseObject = ApplicationWindow() # create a new instance of the application window
    sys.exit(app.exec_()) # exit the python code if the app has received the exit code status

"""
    Execution Entry Point
"""
if __name__ == '__main__':
    execute()