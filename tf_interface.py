"""
    script name: tf_interface.py
    script description: This script is used to interface with the tensorflow model.
"""
import zipfile
import tensorflow as tf
import numpy as np
import utility

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
        print(f"Labels: {self.labels}")

        print("Initialization finished.")

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