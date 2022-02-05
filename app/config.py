"""
    Configuration File for the application.
"""
import pyaudio

FORMAT = pyaudio.paFloat32
CHANNELS = 1
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
CHUNK_MULTIPLIER = 2

MAX_BUFFER_TIME = 5