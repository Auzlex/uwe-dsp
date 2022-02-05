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

SAMPLE_RATES = [
    8000,
    11025,
    16000,
    22050,
    32000,
    44100,
    48000,
    88200,
    96000,
    176400,
    192000
]