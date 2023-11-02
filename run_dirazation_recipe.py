########################################################################

# Author   : David Erik Mollberg & Carlos Daniel Hernández Mena
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.

# ########################################################################
# Step 1
# Convert the audio files and transcript into a diarization format.
# The output is folder for each conversation with three files. Combined
# audio file, and transcript in JSON  and RTTM format
from src.convert2diarization import convert

convert()

# ########################################################################
# Step 2
# To replicate the results for the diarization experiment.
# First install the Pyannote toolkit (https://github.com/pyannote/pyannote-audio)
# per their instructions and your system. Replace the variable with your
# authentication token.
# Using a GPU (device='cuda') the decoding only takes a few minutes.

from src.diraze import diarize

token = ""
# device='cuda'
device = "cpu"
diarize(authentication_token=token, device=device)

# ########################################################################
# Step 3
# To evaluate the results we use the NIST scoring tool, packaged
# here https://github.com/nryant/dscore.
# Install that toolkit and change the variable 'path2dscore' to your
# install location
import subprocess

path2dscore = None
subprocess.call(
    "python",
    f"{path2dscore}/score.py",
    "--collar 0",
    "-R results/diarize/ref_dir.scp",
    "-S results/diarize/hyp_dir.scp",
)
