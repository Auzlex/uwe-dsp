#!/usr/bin/env python3
"""
    import modules
"""
import re
import config # import our custom config file
import pyaudio # to receive audio
import librosa # to extract features
import numpy as np # to handle arrays

class AudioHandler(object):
    """class that handles audio related stuff"""

    def __init__(self) -> None:
        """
            initialize the audio handler
        """
        print( "Initializing audio handler..." )

        # initialize the pyaudio object properties
        self.FORMAT = config.FORMAT
        self.CHANNELS = config.CHANNELS
        self.SAMPLE_RATE = config.SAMPLE_RATE
        self.CHUNK = config.CHUNK_SIZE * config.CHUNK_MULTIPLIER

        # max buffer time in seconds, used to cache data to max n seconds
        self.max_buffer_time = config.MAX_BUFFER_TIME

        # initialize the frame buffer with 0s
        self.frame_buffer = [0] * int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time )
        self.np_buffer = np.array([0] * self.CHUNK * self.max_buffer_time * 10)

        # used by the spectrograph
        self.np_buffer = np.arange(0,0, 2)

         # initialize the pyaudio object
        self.p = pyaudio.PyAudio()#None # reference to pyaudio object
        self.stream = None # reference to pyaudio stream

        print( "Fetching input devices..." )
        self.available_devices_information = self.fetch_input_devices()
        print( "Fetched input devices." )

        print( "Audio handler initialized." )

    # def start(self) -> None:
    #     """start the audio stream"""

    #     print( "Starting audio stream..." )

    #     # initialize the pyaudio object
    #     #self.p = pyaudio.PyAudio()

    #     # open the stream
    #     self.stream = self.p.open(
    #         format=self.FORMAT,
    #         channels=self.CHANNELS,
    #         rate=self.SAMPLE_RATE,
    #         input=True,
    #         stream_callback=self.callback,
    #         frames_per_buffer=self.CHUNK
    #     )

    #     print( "Audio stream started." )

    def start(self, device_index:int=None, format=config.FORMAT, channels=config.CHANNELS, rate=config.SAMPLE_RATE, frames_per_buffer=config.CHUNK_SIZE  ) -> None:
        """start the audio stream"""

        print( f"Starting audio stream..." )

        # open the stream
        self.stream = self.p.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            stream_callback=self.callback,
            frames_per_buffer=frames_per_buffer,
            input_device_index=device_index
        )

        print( "Audio stream started." )

    def stop(self) -> None:
        """stop the audio stream"""
    
        print( "Stopping audio stream..." )

        # close the stream
        self.stream.stop_stream()
        self.stream.close()

        # terminate the pyaudio object
        #self.p.terminate()

        print( "Audio stream stopped." )

    def callback(self, in_data, frame_count, time_info, flag):
        """callback function for the pyaudio stream, returns in np.float32"""

        # get the data from the stream in np.float32 from buffer
        numpy_array = np.frombuffer(in_data, dtype=np.float32) # float32
        
        # append the data to the frame buffer as copy
        self.frame_buffer.append(numpy_array.copy())

        # numpy version of the frame buffer for spectrogram
        self.np_buffer = np.append( self.np_buffer, numpy_array.copy() )
    
        # print(self.np_buffer.ndim, len(self.np_buffer), self.CHUNK * self.max_buffer_time, len(self.np_buffer) > self.CHUNK * self.max_buffer_time)

        # data_np = np.array(numpy_array, dtype='d').flattern()
        # mfcc = librosa.feature.mfcc(self.np_buffer.copy(), sr=self.SAMPLE_RATE)
        # print(mfcc)

        #amplitude = np.frombuffer(in_data,  dtype=np.float32)
        #print(librosa.feature.mfcc(numpy_array))
        #librosa.feature.mfcc(amplitude)
        #print(int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time ))


        #print(len(self.frame_buffer))

        # if the frames list is too long, remove the first element, we only want the last 5 seconds of audio
        if len(self.frame_buffer) > int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time ):
            self.frame_buffer.pop(0)

        # if the np buffer is too large we then check if its greater than the max chunk size we want
        if len(self.np_buffer) > (self.CHUNK) * self.max_buffer_time * 10:
            # delete the first chunk if we exceed max time to track
            self.np_buffer = np.delete(self.np_buffer, list(range(0, self.CHUNK)))

        return None, pyaudio.paContinue

    def is_stream_active(self):
        """returns true if the audio stream is active"""
        if self.stream is None:
            return False

        status = False
        try:
            status = self.stream.is_active()
        except Exception as e:
            print(f"Error checking stream status: {e}")

        return status

    def resample(self, data, original_sr, target_sr):
        """resample the data to the rate"""
        return librosa.resample(data, original_sr, target_sr)

    def fetch_input_devices(self) -> list:
        """
            fetch the input devices
            invokes sd.query_devices(kind="input")
            adjusts the information given to always provide a list of devices in a form of a dict
            
        """
        # get host api information
        info = self.p.get_host_api_info_by_index(0)

        # get the number of devices from the host api
        numdevices = info.get('deviceCount')

        # create a list of devices
        devices = [ self.p.get_device_info_by_host_api_device_index(0, i) for i in range(0, numdevices) if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0 ]
        return devices

    def fetch_supported_sample_rates(self, device_index:int) -> list:

        #print( f"Fetching supported sample rates... for device id {device_index}" )

        supported = []
        devinfo = self.p.get_device_info_by_index(device_index)
        for sr in config.SAMPLE_RATES:
            devinfo = self.p.get_device_info_by_host_api_device_index(0, device_index)

            try:
                #print(f"{devinfo.get('name')} {sr} {self.p.is_format_supported(sr,input_device=devinfo['index'], input_channels=devinfo['maxInputChannels'], input_format=pyaudio.paFloat32)}")
                
                # if the format is supported, add it to the list
                if self.p.is_format_supported(sr,input_device=devinfo['index'], input_channels=devinfo['maxInputChannels'], input_format=pyaudio.paFloat32) is True:
                    supported.append(sr)
            except Exception as e:
                pass
                #print(f"{devinfo.get('name')} {sr} {e}")
            
        return supported   

    def adjust_stream(self, device_index:int, sample_rate:int) -> None:
        """adjust the stream to the device and sample rate"""

        # get the device information
        devinfo = self.p.get_device_info_by_index(device_index)

        # if a stream is active stop it
        if self.is_stream_active():
            self.stop()
        
        # start a new stream on target device
        self.start(
            device_index=device_index,
            rate=sample_rate,
        )


if __name__ == '__main__':

    # new instance of audio handler and runs fetch_input_devices
    audio_handler = AudioHandler()
    for item in audio_handler.available_devices_information:
        print(item)
    #print(audio_handler.fetch_supported_sample_rates())

    # info = audio_handler.p.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')
    # for i in range(0, numdevices):
    #     if (audio_handler.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #         print ("Input Device id ", i, " - ", audio_handler.p.get_device_info_by_host_api_device_index(0, i).get('name'))
    #         print(audio_handler.fetch_supported_sample_rates(i),"\n\n")


    # for device in audio_handler.available_devices_information:
    #     print(f"{device['name']} {device['default_samplerate']} max channels {device['max_input_channels']} {device['hostapi']} :: {device}")

    # print(type(input_device_information) == sd.DeviceList,input_device_information[0])

    # if type(input_device_information) == dict: # we only have 1 device
    #     print(f"only 1 device found {input_device_information['name']} {input_device_information['default_samplerate']} max channels {input_device_information['max_input_channels']}")
    # elif type(input_device_information) == sd.DeviceList:
    #     for device in input_device_information:
    #         print(f"{device['name']} {device['default_samplerate']} max channels {device['max_input_channels']}")

    #print(audio_handler.fetch_input_devices())

# class AudioWavReader(object):

#     def __init__(self, file_path) -> None:
        
#         # load in the libsora file
#         self.y, self.sr = librosa.load(file_path, sr=None)

#         print(self.y, len(self.y))
#         print("\n\n")
#         print(self.sr)


# if __name__ == "__main__":
#     # create an audio handler
#     import os
#     root_dir = os.path.dirname(os.path.realpath(__file__))
#     awr = AudioWavReader(os.path.join(root_dir, "73733292392.wav"))
