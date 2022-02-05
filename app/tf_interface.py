"""
    script name: tf_interface.py
    script description: This script is used to interface with the tensorflow model.
"""
import zipfile
import tensorflow as tf
import numpy as np
import utility

#region layer viewer functions
# def _model2layer(model):
#     """fatch layers name and shape from model"""
#     layers = []

#     for i in model.layers:
#         name = str(i.with_name_scope).split('.')[-1][:-3]
#         if name == 'InputLayer':
#             shape = i.input_shape[0][1:]
#         elif name == 'MaxPooling2D':
#             shape = i.input_shape[1:]
#         else:
#             shape = i.output_shape[1:]
#         layers.append( (tuple(shape), name) )

#     return layers

# def _layer(shape, name):
#     """add more feature on layers"""
#     lay_shape = None
#     lay_name = None
#     lay_color = None
#     lay_marker = None

#     if len(shape) == 1:
#         lay_shape = (shape[0], 1, 1)
#     elif len(shape) == 2:
#         lay_shape = (shape[0], shape[1], 1)
#     else:
#         if name == 'MaxPooling2D' or name == 'AveragePooling2D':
#             lay_shape = (shape[0], shape[1], 1)
#         else:
#             lay_shape = shape

#     lay_name = name

#     if len(lay_shape) == 3 and lay_shape[-1] == 3:
#         lay_color = 'rgb'
#         lay_marker = 'o'
#     else:
#         if lay_name == 'InputLayer':
#             lay_color = 'r'
#             lay_marker = 'o'
#         elif lay_name == 'Conv2D':
#             lay_color = 'y'
#             lay_marker = '^'
#         elif lay_name == 'MaxPooling2D' or lay_name == 'AveragePooling2D':
#             lay_color = 'c'
#             lay_marker = '.'
#         else:
#             lay_color = 'g'
#             lay_marker = '.'

#     return {'shape': lay_shape, 'name': lay_name, 'color': lay_color, 'marker': lay_marker}

# def _shape2array(shape, layers_len, xy_max):
#     """create shape to array/matrix"""

#     shape = np.asarray(shape)

#     x = shape[0]
#     y = shape[1]
#     z = shape[2]

#     single_layer = []

#     if xy_max[0] < x:
#         xy_max[0] = x
#     if xy_max[1] < y:
#         xy_max[1] = y

#     for k in range(z):
#         arr_x, arr_y, arr_z = [], [], []

#         for i in range(y):
#             ox = [j for j in range(x)]
#             arr_x.append(ox)

#         for i in range(y):
#             oy = [j for j in (np.ones(x, dtype=int) * i)]
#             arr_y.append(oy)

#         for i in range(y):
#             oz = [j for j in (np.ones(y, dtype=int) * layers_len)]
#             arr_z.append(oz)

#         layers_len += 2
#         single_layer.append([arr_x, arr_y, arr_z])

#     layers_len += 4

#     return single_layer, layers_len, xy_max


#             # if self.connection:
#             #     if name == 'Dense' or name == 'Flatten':
#             #         for c in line_z:
#             #             a, b, c = line_x[0], line_y[0], c
#             #             if temp:
#             #                 temp = False
#             #                 last_a, last_b, last_c = a, b, c
#             #                 continue

#             #             if color_in == 'rgb':
#             #                 color = color_in[color_count]
#             #                 color_count += 1

#             #             else:
#             #                 color = color_in

#             #             self._dense(ax, a[0], a[1], b[0], b[1], last_a[0], last_a[1], last_b[0], last_b[1], c[0],
#             #                         last_c[0], c=color)
#             #             last_a, last_b, last_c = a, b, c
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
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.summary()

    def predict(self, image):
        """
            Method name: predict
            Method description: This method is used to predict the image.
            Input:
                image: The image to predict.
            Output:
                The prediction.
        """
        return self.model.predict(image)

    def predict_class(self, image):
        """
            Method name: predict_class
            Method description: This method is used to predict the class of the image.
            Input:
                image: The image to predict.
            Output:
                The prediction.
        """
        return np.argmax(self.predict(image))

    def predict_class_name(self, image):
        """
            Method name: predict_class_name
            Method description: This method is used to predict the class name of the image.
            Input:
                image: The image to predict.
            Output:
                The prediction.
        """
        return utility.get_class_name(self.predict_class(image))

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