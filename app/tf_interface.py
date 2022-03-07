"""
    script name: tf_interface.py
    script description: This script is used to interface with the tensorflow model.
"""
import zipfile
import tensorflow as tf
import numpy as np
import utility
import h5py
import json
import config
from keras.models import load_model

def load_model_ext(filepath, custom_objects=None):
    model = load_model(filepath, custom_objects=None)
    f = h5py.File(filepath, mode='r')

    meta_data = None
    if 'metadata' in f.attrs: # metadata
        meta_data = f.attrs.get('metadata')

    dfe = None
    if 'dfe' in f.attrs: # desired feature extraction
        dfe = f.attrs.get('dfe')

    f.close()
    return model, meta_data, dfe

#endregion
class TFInterface:
    """
        Class name: TFInterface
        Class description: This class is used to interface with the tensorflow model.
    """
    def __init__(self, model_path):
        """
            Method name: __init__
            Method description: This method is used to initialize the TFInterface class.
            Input:
                model_path: The path to the model.
        """
        self.model_path = model_path
        self.model, self.metadata, self.dfe = load_model_ext(model_path)
        if self.metadata is not None:
            self.metadata = json.loads(self.metadata)#json.loads('["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]')#json.loads(self.metadata)
        
        self.layers = self._model2layers()

        print(f"model dfe: {self.dfe}")
        print(self.model.layers[0])


    def _model2layers(self):
        """fatch layers name and shape from model"""
        layers = []

        for i in self.model.layers:
            name = str(i.with_name_scope).split('.')[-1][:-3]
            if name == 'InputLayer':
                shape = i.input_shape[0][1:]
            elif name == 'MaxPooling2D':
                shape = i.input_shape[1:]
            else:
                shape = i.output_shape[1:]
            layers.append((tuple(shape), name))

        return layers

    def predict_multi_mfcc(self, mffc_data:list):
        # n_mfcc = 40
        # sampling_rate = 44100
        # audio_duration = 4
        # audio_length = audio_duration * sampling_rate
        # input_shape = (n_mfcc, 517, 1) # 1 + int(np.floor(audio_length/512))

        # array = np.resize(mffc_data, input_shape)
        # array = array.reshape(1, array.shape[0], array.shape[1], array.shape[2])

        predictions = self.model.predict([mffc_data])
        return predictions
        # index = np.argmax(prediction, axis=None, out=None)

        # if self.metadata is not None:

        #     #print(index, self.metadata[index], type(self.metadata) )
        #     return self.metadata[index]
        # else:
        #     return np.argmax(prediction)

    def predict_formatted_data(self, data):
        """
            Method name: predict
            Method description: This method is used to predict an audio sample.
            Input:
                audio_sample: normalized audio sample of 2 seconds.
            Output:
                The prediction.
        """

        # # audio_duration = 4
        # # mffc_data = mffc_data[0:config.CHUNK_SIZE * config.SAMPLE_RATE * audio_duration]

        # # n_mfcc = 40#128#40
        # # sampling_rate = 44100
        # # audio_length = audio_duration * sampling_rate
        # # input_shape = (n_mfcc, 1 + int(np.floor(audio_length/512)), 1)

        # n_mfcc = 40
        # # sampling_rate = 44100
        # # audio_duration = 4
        # # audio_length = audio_duration * sampling_rate
        # input_shape = (n_mfcc, 517, 1) # 1 + int(np.floor(audio_length/512))

        # # pad the mffcc_data array to the input shape
        # array = np.pad(mffc_data, (0, input_shape[1] - mffc_data.shape[0]), 'constant')
        
        # array = np.resize(array, input_shape)
        # array = array.reshape(1, array.shape[0], array.shape[1], array.shape[2])
        # #array = mffc_data.resize(input_shape, refcheck=False) #np.resize(mffc_data, input_shape)
        # #array = array.reshape(1, array.shape[0], array.shape[1], array.shape[2])#array.reshape( 1, *self.model.layers[0].input_shape )#array.reshape(1, array.shape[0], array.shape[1], array.shape[2])

        predictions = []
        # #self.model.layers[0].input_shape
        for segment in data:
            prediction = self.model.predict([segment])
            predictions.append(prediction)

        return predictions

        # index = np.argmax(prediction, axis=None, out=None)

        # if self.metadata is not None:

        #     #print(index, self.metadata[index], type(self.metadata) )
        #     return self.metadata[index]
        # else:
        #     return np.argmax(prediction)

    def predict_mel(self, mel_data):
        """
            Method name: predict
            Method description: This method is used to predict an audio sample.
            Input:
                audio_sample: normalized audio sample of 2 seconds.
            Output:
                The prediction.
        """

        audio_duration = 4
        #mffc_data = mffc_data[0:config.CHUNK_SIZE * config.SAMPLE_RATE * audio_duration]

        # n_mfcc = 128#40
        # sampling_rate = 44100
        # audio_length = audio_duration * sampling_rate
        # input_shape = (n_mfcc, 1 + int(np.floor(audio_length/512)), 1)

        n_mfcc = 40
        sampling_rate = 44100
        audio_duration = 4
        audio_length = audio_duration * sampling_rate
        input_shape = (n_mfcc, 517, 1) # 1 + int(np.floor(audio_length/512))

        array = np.resize(mel_data, input_shape)
        array = array.reshape(1, array.shape[0], array.shape[1], array.shape[2])

        prediction = self.model.predict([array])

        index = np.argmax(prediction, axis=None, out=None)

        if self.metadata is not None:

            #print(index, self.metadata[index], type(self.metadata) )
            return self.metadata[index]
        else:
            return np.argmax(prediction)

    # def predict_class(self, image):
    #     """
    #         Method name: predict_class
    #         Method description: This method is used to predict the class of the image.
    #         Input:
    #             image: The image to predict.
    #         Output:
    #             The prediction.
    #     """
    #     return np.argmax(self.predict(image))

    # def predict_class_name(self, image):
    #     """
    #         Method name: predict_class_name
    #         Method description: This method is used to predict the class name of the image.
    #         Input:
    #             image: The image to predict.
    #         Output:
    #             The prediction.
    #     """
    #     return utility.get_class_name(self.predict_class(image))

class TFLiteInterface:
    """
        Class name: TFLiteInterface
        Class description: This class is used to interface with the tensorflow lite model.
    """
    def __init__(self, model_path):
        """
            Method name: __init__
            Method description: This method is used to initialize the TFLiteInterface class.
            Input:
                model_path: The path to the model.
        """
        print(f"Initializing TensorFlow Lite Model: {str(model_path)}")
        self.model_path = model_path
        self.model = tf.lite.Interpreter(model_path=self.model_path)

        input_details = self.model.get_input_details()
        self.waveform_input_index = input_details[0]['index']
        output_details = self.model.get_output_details()
        self.scores_output_index = output_details[0]['index']

        self.resize_tensor_input(37152)

        print(self.waveform_input_index, self.scores_output_index)

        self.model.allocate_tensors()


        # setup label references
        labels_file = zipfile.ZipFile(self.model_path).open('yamnet_label_list.txt')
        self.labels = [l.decode('utf-8').strip() for l in labels_file.readlines()]
        print(f"Labels: {self.labels}, {len(self.labels)}")

        print("Initialization finished.")

        #print(self.model.get_tensor_details())
        print("\n\n")
        #print(_model2layer(self.model))

        self.layers_len = 0
        self.layers_array = []
        self.layers_name = []
        self.layers_marker = []
        self.layers_color = []
        self.xy_max = [0, 0]

        # #https://stackoverflow.com/questions/62276989/tflite-can-i-get-graph-layer-sequence-information-directly-through-the-tf2-0-a
        # for layer in self.model.get_tensor_details():
        #     #print(layer['shape'],'\t',layer['name'])

        #     #list_ = list(map(lambda x: x, layer['shape']))

        #     if len(layer['shape']) > 0:
        #         #print(list(map(lambda x: x, layer['shape']))[0])

        #         layer_dict = _layer(layer['shape'], layer['name'])
        #         single_layer, layers_len, xy_max = _shape2array(layer_dict['shape'], self.layers_len, self.xy_max)

        #         self.layers_len = layers_len
        #         self.xy_max = xy_max

        #         self.layers_array.append(single_layer)
        #         self.layers_name.append(layer_dict['name'])
        #         self.layers_color.append(layer_dict['color'])
        #         self.layers_marker.append(layer_dict['marker'])

        # print(len(self.layers_array))

            # print("\nLayer Name: {}".format(layer['name']))
            # print("\tIndex: {}".format(layer['index']))
            # print("\n\tShape: {}".format(layer['shape']))
            # print("\tTensor: {}".format(self.model.get_tensor(layer['index']).shape))
            # print("\tTensor Type: {}".format(self.model.get_tensor(layer['index']).dtype))
            # print("\tQuantisation Parameters")
            # print("\t\tScales: {}".format(layer['quantization_parameters']['scales'].shape))
            # print("\t\tScales Type: {}".format(layer['quantization_parameters']['scales'].dtype))
            # print("\t\tZero Points: {}".format(layer['quantization_parameters']['zero_points']))
            # print("\t\tQuantized Dimension: {}".format(layer['quantization_parameters']['quantized_dimension']))

    def resize_tensor_input(self, waveform_size:int):
        self.model.resize_tensor_input(self.waveform_input_index, [waveform_size], strict=False)

    def feed(self, waveform):
        self.model.set_tensor(self.waveform_input_index, waveform)
        self.model.invoke()

    def fetch_best_score_index(self):
        scores = self.model.get_tensor(self.scores_output_index)
        #top_class_index = scores.argmax()
        return scores.argmax()

    # def predict(self, image):
    #     """
    #         Method name: predict
    #         Method description: This method is used to predict the image.
    #         Input:
    #             image: The image to predict.
    #         Output:
    #             The prediction.
    #     """
    #     input_details = self.model.get_input_details()
    #     output_details = self.model.get_output_details()
    #     self.model.set_tensor(input_details[0]['index'], image)
    #     self.model.invoke()
    #     return self.model.get_tensor(output_details[0]['index'])

    # def predict_class(self, image):
    #     """
    #         Method name: predict_class
    #         Method description: This method is used to predict the class of the image.
    #         Input:
    #             image: The image to predict.
    #         Output:
    #             The prediction.
    #     """
    #     return np.argmax(self.predict(image))

    # def predict_class_name(self, image):
    #     """
    #         Method name: predict_class_name
    #         Method description: This method is used to predict the class name of the image.
    #         Input:
    #             image: The image to predict.
    #         Output:
    #             The prediction.
    #     """
    #     return utility.get_class_name(self.predict_class(image))

if __name__ == "__main__":

    tf_interface = TFInterface("/home/charlesedwards/Documents/kaggle_2018_dataset/models/CC trained/K2018_MFCC_RESNET32_lr-5e-06_b1-0.99_b2-0.999_EPOCH-500_BATCH-32_categorical_crossentropy.h5")

    # parent = np.empty()
    # np.append( parent, [ 0, 0, 0 ] )

    # array = np.zeros((4, 3))
    # print(array)