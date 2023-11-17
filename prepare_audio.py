########################################################################

# Author   : David Erik Mollberg & Carlos Daniel Hernández Mena
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.


# Prepare the audiofiles for ASR and diarization without running the
# full recepies.
########################################################################


SPLITS_FOLDER = "splits"
SEGMENTED = "segmented"

print("Preparing audiofiles")

# Convert the audiofiles into smaller segments of 2-20 seconds.
from src.segment import run_segmentation

_, _, _ = run_segmentation(SEGMENTED, SPLITS_FOLDER, min_duration=2, max_duration=20)

# Convert the audio files into the diarization format.
# The output is folder for each conversation with three files. Combined
# audio file, and transcript in JSON  and RTTM format
from src.convert2diarization import convert

convert()
