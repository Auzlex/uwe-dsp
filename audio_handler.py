#!/usr/bin/env python3
"""
    import modules
"""
import config # import our custom config file
import simpleaudio as sa # to play audio
import pyaudio # to receive audio
import librosa # to extract features
import numpy as np # to handle arrays
import time # to handle time

class AudioHandler(object):
    """class that handles audio related stuff"""

    def __init__(self):
        """initialize the audio handler"""

        print( "Initializing audio handler..." )

        # initialize the pyaudio object properties
        self.FORMAT = config.FORMAT
        self.CHANNELS = config.CHANNELS
        self.SAMPLE_RATE = config.SAMPLE_RATE
        self.CHUNK = config.CHUNK_SIZE * config.CHUNK_MULTIPLIER

        self.max_buffer_time_in_seconds = config.MAX_BUFFER_TIME_IN_SECONDS

        # initialize the frame buffer with 0s
        self.frame_buffer = [0] * int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time_in_seconds )#np.arange(0, 2 * self.CHUNK, 2)#[0] * int(self.SAMPLE_RATE / self.CHUNK * 10)#[0] * int(self.SAMPLE_RATE / self.CHUNK * 10)
        self.raw_frame_buffer = []#[0] * int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time_in_seconds )

        self.p = None # reference to pyaudio object
        self.stream = None # reference to pyaudio stream

        print( "Audio handler initialized." )

    def start(self):
        """start the audio stream"""

        print( "Starting audio stream..." )

        # initialize the pyaudio object
        self.p = pyaudio.PyAudio()

        # open the stream
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            stream_callback=self.callback,
            frames_per_buffer=self.CHUNK
        )

        print( "Audio stream started." )


    def stop(self):
        """stop the audio stream"""
    
        print( "Stopping audio stream..." )

        # close the stream
        self.stream.stop_stream()
        self.stream.close()

        # terminate the pyaudio object
        self.p.terminate()

        print( "Audio stream stopped." )

    def callback(self, in_data, frame_count, time_info, flag):
        """callback function for the pyaudio stream, returns in np.float32"""

        numpy_array = np.frombuffer(in_data, dtype=np.float32) # float32
        #amplitude = np.frombuffer(in_data,  dtype=np.float32)
        #print(librosa.feature.mfcc(numpy_array))
        #librosa.feature.mfcc(amplitude)

        self.frame_buffer.append(numpy_array)

        self.raw_frame_buffer = np.concatenate([self.raw_frame_buffer, numpy_array])
        #print(self.raw_frame_buffer, len(self.raw_frame_buffer))
        #print("\n\n")
        #print(librosa.feature.mfcc(self.raw_frame_buffer.copy(),sr=self.SAMPLE_RATE))
        
        #print(librosa.feature.mfcc(self.raw_frame_buffer,sr=self.SAMPLE_RATE))
        #self.raw_frame_buffer.append(librosa.feature.mfcc(numpy_array,sr=self.SAMPLE_RATE))
        #print(numpy_array)

        # if the frames list is too long, remove the first element, we only want the last 5 seconds of audio
        
        #print(int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time_in_seconds ))
        if len(self.frame_buffer) > int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time_in_seconds ):
            self.frame_buffer.pop(0)

        if len(self.raw_frame_buffer) > int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time_in_seconds ):
            #self.raw_frame_buffer.pop(0)
            self.raw_frame_buffer = self.raw_frame_buffer[1:int(self.SAMPLE_RATE / self.CHUNK * self.max_buffer_time_in_seconds )]
            

        return None, pyaudio.paContinue

    def is_stream_active(self):
        """returns true if the audio stream is active"""
        return self.stream.is_active()

    def mainloop(self):
        """main loop for the audio handler"""

        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(2.0)



class AudioWavReader(object):

    def __init__(self, file_path) -> None:
        
        # load in the libsora file
        self.y, self.sr = librosa.load(file_path, sr=None)

        print(self.y)
        print("\n\n")
        print(self.sr)


if __name__ == "__main__":
    # create an audio handler
    import os
    root_dir = os.path.dirname(os.path.realpath(__file__))
    awr = AudioWavReader(os.path.join(root_dir, "73733292392.wav"))
