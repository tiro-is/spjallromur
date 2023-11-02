########################################################################

# Author   : Carlos Daniel Hernández Mena & David Erik Mollberg
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.

# Description:

# This is script downloads the Faster-Whisper model:
# "language-and-voice-lab/whisper-large-icelandic-30k-steps-1000h-ct2"
# From Hugging Face.

########################################################################

import os

HF_ASR_MODEL = "https://huggingface.co/language-and-voice-lab/whisper-large-icelandic-30k-steps-1000h-ct2"

########################################################################


def download_asr_model():
    CURRENT_PATH = os.getcwd()
    NAME_ASR_MODEL = os.path.basename(HF_ASR_MODEL)
    PATH_ASR_MODEL = os.path.join(CURRENT_PATH, NAME_ASR_MODEL)
    if not os.path.exists(PATH_ASR_MODEL):
        os.system("git clone " + HF_ASR_MODEL)
    # ENDIF
    return PATH_ASR_MODEL


# ENDDEF

########################################################################
