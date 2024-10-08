import sys
import os

import numpy as np

sys.path.append('C:/Users/user/Desktop/ASP_Coursera/sms-tools-master/software/models')
from utilFunctions import wavread

"""
A1-Part-2: Basic operations with audio

Write a function that reads an audio file and returns the minimum and the maximum values of the audio 
samples in that file. 

The input to the function is the wav file name (including the path) and the output should be two floating 
point values returned as a tuple.

If you run your code using oboe-A4.wav as the input, the function should return the following output:  
(-0.83486432, 0.56501967)
"""


def minMaxAudio(inputFile):
    """
    Input:
        inputFile: file name of the wav file (including path)
    Output:
        A tuple of the minimum and the maximum value of the audio samples, like: (min_val, max_val)
    """
    # Your code here

    (fs, x) = wavread(inputFile)

    minValue = float(min(x))
    maxValue = float(max(x))

    return minValue, maxValue
