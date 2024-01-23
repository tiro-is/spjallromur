########################################################################

# Author   : Carlos Daniel Hernández Mena & David Erik Mollberg
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.

# Description:

# This scripts tries to import all the python libraries needed to
# run the recipe. It also checks the existance of ffmpeg in the system.

# If you see no error messages after running this script, then you
# will be able to successfully run the recipe in the current system.

########################################################################


def check_dependencies():
    ########################################################################
    # Important Variables

    ERROR_FLAG = False

    ########################################################################
    # Imports

    import json
    import os
    import re
    import shutil
    import sys
    import wave

    import requests
    import tqdm

    ########################################################################

    try:
        import jiwer
    except:
        print('\nERROR: You need to install the library "jiwer" in your system.')
        print("In your conda environment try with:")
        print("\n\t$ pip install jiwer\n")
        ERROR_FLAG = True
    # ENDTRY

    ########################################################################

    try:
        from faster_whisper import WhisperModel
        from recipe_scripts.transcribe_splits import transcribe_splits

    except:
        print(
            '\nERROR: You need to install the library "faster-whisper" in your system.'
        )
        print("In your conda environment try with:")
        print("\n\t$ pip install faster-whisper\n")
        ERROR_FLAG = True
    # ENDTRY

    ########################################################################
    # Check the existance of ffmpeg in the system
    import subprocess

    cmd = "ffmpeg -version | head -n 1"
    shell_out = subprocess.check_output(cmd, shell=True)

    shell_out = shell_out.decode("utf-8")
    shell_out = re.sub("\s+", " ", shell_out)
    shell_out = shell_out.strip()
    list_shell = shell_out.split(" ")

    first_word = list_shell[0]
    first_word_ok = "ffmpeg"

    if first_word != first_word_ok:
        print('\nERROR: You need to install the command "ffmpeg" in your system.')
        print("In Ubuntu try with:")
        print("\n\t$ sudo apt install ffmpeg\n")
        ERROR_FLAG = True
    # ENDIF

    ########################################################################
    # Check the existance of git-lfs in the system

    cmd = "git-lfs -v"
    shell_out = subprocess.check_output(cmd, shell=True)

    shell_out = shell_out.decode("utf-8")
    shell_out = shell_out[0:7]

    first_word = shell_out
    first_word_ok = "git-lfs"

    if first_word != first_word_ok:
        print('\nERROR: You need to install the command "git-lfs" in your system.')
        print("In Ubuntu try with:")
        print("\n\t$ sudo apt install git-lfs\n")
        ERROR_FLAG = True
    # ENDIF

    ########################################################################
    # Check the existance of sox in the system

    cmd = "sox --version"
    shell_out = subprocess.check_output(cmd, shell=True)

    shell_out = shell_out.decode("utf-8")
    shell_out = shell_out[0:3]

    first_word = shell_out
    first_word_ok = "sox"

    if first_word != first_word_ok:
        print('\nERROR: You need to install the command "sox" in your system.')
        print("In Ubuntu try with:")
        print("\n\t$ sudo apt install sox\n")
        ERROR_FLAG = True
    # ENDIF

    ########################################################################

    if ERROR_FLAG == False:
        print("\n\tSuccess: Your system is ready to run the recipe!\n")
    # ENDIF

    ########################################################################
