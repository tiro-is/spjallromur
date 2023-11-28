########################################################################

# Author   : Carlos Daniel Hernández Mena & David Erik Mollberg
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.


# This script provides utilities for merging audio files and their corresponding transcripts.
# It is designed to work with two separate speaker files and combine them into a single
# stereo audio and a unified transcript. It also converts the combined transcript into the
# RTTM (Rich Transcription Time Marked) format.

########################################################################

import json
import os
import subprocess
from glob import glob

# Define paths
ROOT = "."
OUTPUT_ROOT = "combined"


def merge_audio_files(file1, file2, output_file):
    """
    Merges two single-channel WAV files into a single two-channel WAV file.

    Args:
        file1 (str): Path to the first audio file.
        file2 (str): Path to the second audio file.
        output_file (str): Path where the merged file will be saved.
    """
    command = ["sox", "-M", file1, file2, "-c1", output_file]

    try:
        subprocess.run(command, check=True)
        print(f"Files merged successfully into {output_file}")
    except subprocess.CalledProcessError:
        print("Error merging audio files.")


def merge_transcripts(transcript_a, transcript_b, output_file):
    """
    Combines the transcripts of two speakers into a single JSON file.

    Args:
        transcript_a (str): Path to the transcript of the first speaker.
        transcript_b (str): Path to the transcript of the second speaker.
        output_file (str): Path where the combined transcript will be saved.

    Returns:
        dict: The combined transcript data.
    """
    with open(transcript_a, "r") as file:
        trans_a = json.load(file)
    with open(transcript_b, "r") as file:
        trans_b = json.load(file)

    for w in trans_a["words"]:
        w["spk"] = "a"
    for w in trans_b["words"]:
        w["spk"] = "b"

    combined = trans_a["words"] + trans_b["words"]
    combined.sort(key=lambda s: s["start"])

    data = {
        "metadata": {
            "speaker_a": {
                "age": trans_a["metadata"]["age"],
                "gender": trans_a["metadata"]["gender"],
            },
            "speaker_b": {
                "age": trans_b["metadata"]["age"],
                "gender": trans_b["metadata"]["gender"],
            },
        },
        "words": combined,
    }
    with open(output_file, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return data


def convert2rttm(data, output_file, file_id):
    """
    Converts the combined transcript into the RTTM format.

    Args:
        data (dict): The combined transcript data.
        output_file (str): Path where the RTTM file will be saved.
        file_id (str): A unique identifier for the audio file.
    """
    rttm = []
    spk, curr_spk = data["words"][0]["spk"], data["words"][0]["spk"]
    start = data["words"][0]["start"]
    spk_mapping = {
        "a": f"a_{data['metadata']['speaker_a']['age']}_{data['metadata']['speaker_a']['gender']}",
        "b": f"b_{data['metadata']['speaker_b']['age']}_{data['metadata']['speaker_b']['gender']}",
    }
    duration = 0.1
    for idx, word in enumerate(data["words"]):
        spk = word["spk"]
        if idx > 0:
            duration = round(data["words"][idx - 1]["end"] - start, 2)

        if duration == 0.0:
            print("stop")
        if curr_spk != spk:
            s = f"SPEAKER {file_id} 1 {start} {duration} <NA> <NA> {spk_mapping[spk]} <NA> <NA>"
            rttm.append(s)

            curr_spk = spk
            start = word["start"]

        # Add the last turn
        if idx == len(data["words"]) - 1:
            duration = round(word["end"] - start, 2)
            if duration > 0.2:
                s1 = f"SPEAKER {file_id} 1 {start} {duration} <NA> <NA> {spk_mapping[spk]} <NA> <NA>"
                rttm.append(s1)

    with open(output_file, "w") as f_out:
        f_out.write("\n".join(rttm) + "\n")


def convert():
    """
    Main execution function.
    Merges speaker transcripts and audio files, then converts the transcript to RTTM format.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for folder in glob(f"{ROOT}/full_conversations/*"):
        print(f"Preparing '{folder}'")
        spk_a_trans = glob(f"{folder}/a*.json")[0]
        spk_b_trans = glob(f"{folder}/b*.json")[0]
        spk_a_audio = glob(f"{folder}/a*.wav")[0]
        spk_b_audio = glob(f"{folder}/b*.wav")[0]

        folder_base = os.path.basename(folder)
        new_filename = f"combined_{folder_base}"
        out_dir = os.path.join(OUTPUT_ROOT, folder_base)
        os.makedirs(out_dir, exist_ok=True)

        merged_json = os.path.join(out_dir, f"{new_filename}.json")
        merged_rttm = os.path.join(out_dir, f"{new_filename}.rttm")

        if any([os.path.exists(merged_json), os.path.exists(merged_rttm)]):
            print(f"{merged_json} or {merged_rttm} exist, wont overwrite.")
            continue
        data = merge_transcripts(
            spk_a_trans,
            spk_b_trans,
            merged_json,
        )

        convert2rttm(data, merged_rttm, new_filename)
        merge_audio_files(
            spk_a_audio, spk_b_audio, os.path.join(out_dir, f"{new_filename}.wav")
        )


if __name__ == "__main__":
    convert()
