########################################################################

# Author   : David Erik Mollberg & Carlos Daniel Hernández Mena
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.

# Description:

# This script provides functionality to automatically diarize audio recordings.
# Additionally, the script generates SCP files, containing paths to RTTM
# files for both reference and hypothesis diarization results.

########################################################################


import os
from glob import glob

import torch
from pyannote.audio import Pipeline
from tqdm import tqdm

AUDIO_DATA_ROOT = "combined"
DIR_RESULTS = "results/diarize/rttm"

HYP_SCP = "results/diarize/hyp_dir.scp"
REF_SCP = "results/diarize/ref_dir.scp"


def diarize(authentication_token: str, device: str = "cpu"):
    """
    Diarizes audio recordings in the specified root directory.

    Parameters:
    - authentication_token (str): Token for authenticating with the Hugingface Hub.
    - device (str, optional): Device to which the pipeline model is sent. Defaults to 'cpu'.

    Output:
    This function will create RTTM files in the specified results directory and
    SCP files containing paths to the RTTM files for both reference and hypothesis.
    """

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=authentication_token,
    )

    pipeline.to(torch.device(device))
    os.makedirs(DIR_RESULTS, exist_ok=True)

    rttm_reference_scp = []
    rttm_hypothesis_scp = []

    for folder in tqdm(glob(f"{AUDIO_DATA_ROOT}/*")):
        folder_name = os.path.basename(folder)
        audio_file = glob(f"{folder}/*.wav")[0]
        diarization = pipeline(audio_file, num_speakers=2)

        hypothesis_rttm_file = os.path.join(DIR_RESULTS, folder_name + "_2.rttm")
        with open(hypothesis_rttm_file, "w") as rttm:
            diarization.write_rttm(rttm)

        rttm_reference_file = glob(f"{folder}/*.rttm")[0]
        rttm_reference_scp.append(rttm_reference_file)
        rttm_hypothesis_scp.append(hypothesis_rttm_file)

    with open(HYP_SCP, "w") as f:
        f.write("\n".join(rttm_hypothesis_scp))
    with open(REF_SCP, "w") as f:
        f.write("\n".join(rttm_reference_scp))
