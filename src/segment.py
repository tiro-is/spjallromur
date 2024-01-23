########################################################################

# Author   : David Erik Mollberg & Carlos Daniel Hernández Mena
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.

# Module to segment audio data from the Spjallromur dataset.

########################################################################

import json
import os
import re
import subprocess
from glob import glob
from typing import Tuple


def compile_files() -> list:
    """
    Retrieve all audio file paths in given root directory.

    Parameters:
    - root (str): The root directory to search in.

    Returns:
    - list: List of audio file paths.
    """
    paths = []
    if os.path.exists("full_conversations"):
        paths += [x for x in glob("full_conversations/*/*.wav")]
        print("Using full coversations")

    if os.path.exists("half_conversations"):
        paths += [x for x in glob("half_conversations/*/*.wav")]
        print("Using half coversations")

    if paths:
        return paths
    else:
        raise Exception("No audio files found")


def extract_audio_segment(
    input_file: str,
    output_file: str,
    start_time: float,
    duration: float,
    overwrite: bool = False,
):
    """
    Extract a segment from an audio file using the SoX tool.

    Parameters:
    - input_file (str): Path to the source audio file.
    - output_file (str): Path to the destination audio file.
    - start_time (float): Start time of the segment to extract (in seconds).
    - duration (float): Duration of the segment to extract (in seconds).
    - overwrite (bool): Whether to overwrite the output file if it already exists.
    """

    if not overwrite and os.path.exists(output_file):
        return
    try:
        command = [
            "sox",
            input_file,
            output_file,
            "trim",
            str(start_time),
            str(duration),
        ]
        subprocess.run(command, check=True)
        print(f"Success: Audio segment saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_transcript(file: str) -> list:
    """
    Load the transcript from a given JSON file.

    Parameters:
    - file (str): Path to the JSON file.

    Returns:
    - list: List of word objects extracted from the transcript.
    """

    data = json.load(open(file))["words"]
    return data


def flatten(seg: list) -> dict:
    """
    Convert a list of word objects into a single text segment with timestamps.

    Parameters:
    - seg (list): List of word objects.

    Returns:
    - dict: Flattened segment information.
    """
    return {
        "text": " ".join([x["word"] for x in seg]),
        "text_norm": " ".join([x["norm_word"] for x in seg]),
        "start": seg[0]["start"],
        "end": seg[-1]["end"],
        "duration": round(seg[-1]["end"] - seg[0]["start"], 2),
    }


def open_splits_files(splits_folder: str) -> dict:
    """
    Load and map the split information for audio files.

    Parameters:
    - splits_folder (str): Folder containing split information.

    Returns:
    - dict: Mapping from file_id to its respective split.
    """

    mapping = {}
    for s in ["dev", "train", "test"]:
        f = os.path.join(splits_folder, s)
        if not os.path.exists(f):
            raise Exception(f"File {f} does not exist")
        mapping.update({x.rstrip(): s for x in open(f)})
    return mapping


def segment_duration(seg: list) -> float:
    """
    Calculate the duration of a given segment.

    Parameters:
    - seg (list): Segment to calculate the duration for.

    Returns:
    - float: Duration of the segment.
    """

    return round(seg[-1]["end"] - seg[0]["start"], 2)


def split_in_half(seg: list) -> Tuple[list, list]:
    """
    Split a segment roughly in half by time.

    Parameters:
    - seg (list): Segment to be split.

    Returns:
    - Tuple[list, list]: Two new segments after splitting.
    """

    duration = segment_duration(seg)
    middle_timestamp = seg[0]["start"] + duration / 2

    seg_a, seg_b = [], []
    for s in seg:
        if s["end"] <= middle_timestamp:
            seg_a.append(s)
        else:
            seg_b.append(s)
    return seg_a, seg_b


def add_padding(curr_end: dict, next_end: float) -> list:
    """
    Add padding to the end of segments if possible to
    to account for potentail inaccuracies in the timestamps.
    We add 0.5 seconds padding to the end of each segment
    where that dosent cross over to the next segment.

    Parameters:
    - curr_end (dict): Current segment information.
    - next_end (float): Ending timestamp of the next segment.

    Returns:
    - list: End timestamp with added padding.
    """

    if next_end - curr_end > 0.5:
        end_padding = curr_end + 0.5
    else:
        end_padding = (
            curr_end + next_end - curr_end - 0.1
            if curr_end + next_end - curr_end - 0.1 < next_end
            else curr_end + next_end - curr_end
        )

    if end_padding < 0:
        input("Press enter to continue")
    return end_padding


def merge_segments(seg1: list, seg2: list) -> list:
    """
    Merge two segments into a single segment.

    Parameters:
    - seg1 (list): First segment.
    - seg2 (list): Second segment.

    Returns:
    - list: Merged segment.
    """
    return {
        "text": re.sub("\s+", " ", " ".join([seg1["text"], seg2["text"]])),
        "text_norm": re.sub(
            "\s+", " ", " ".join([seg1["text_norm"], seg2["text_norm"]])
        ),
        "start": seg1["start"],
        "end": seg2["end"],
        "duration": round(seg2["end"] - seg1["start"], 2),
    }


def run_segmentation(
    output_folder: str,
    splits_folder: str,
    min_duration: int = 2,
    max_duration: int = 20,
) -> Tuple[str, str, str]:
    """
    Create short segments for each recording in the corpus.

    Parameters:
    - output_folder (str): Directory to save segmented audio and its transcripts.
    - splits_folder (str): Folder containing split information.
    - min_duration (int): Minimum acceptable segment duration. Default is 2 seconds.
    - max_duration (int): Maximum acceptable segment duration. Default is 20 seconds.

    Returns:
    - Tuple[str, str, str]: Paths to the transcript files for dev, test, and train splits.
    """

    if not os.path.exists(splits_folder):
        raise Exception(f"Folder {splits_folder} does not exist")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for s in ["dev", "train", "test"]:
        f = os.path.join(splits_folder, s)
        if not os.path.exists(f):
            raise Exception(f"File {f} does not exist")
        else:
            os.makedirs(os.path.join(output_folder, s), exist_ok=True)

    fileId2split = open_splits_files(splits_folder)

    info = []
    for audio_file in compile_files():
        file_id = os.path.basename(audio_file).rstrip(".wav")
        print(audio_file.replace(".wav", ".json"))
        transcript = load_transcript(audio_file.replace(".wav", ".json"))

        segments = []
        segment = []

        for idx, item in enumerate(transcript):
            segment.append(item)
            if segment and item["word"][-1] in [".", "?", "!"]:
                if idx + 1 < len(transcript):
                    segment[-1]["end"] = add_padding(
                        segment[-1]["end"], transcript[idx + 1]["end"]
                    )
                segments.append(segment)
                segment = []

        if segment:
            segments.append(segment)

        total_segments_before = len(segments)
        iterations = 1
        while [seg for seg in segments if segment_duration(seg) > max_duration]:
            print(
                f"- Found {len([seg for seg in segments if segment_duration(seg) > max_duration])} segments that are too long."
            )
            idx = 0
            while idx < len(segments):
                seg = segments[idx]

                if segment_duration(seg) > max_duration:
                    print(f"-- Found segment that is {segment_duration(seg)}")
                    a, b = split_in_half(seg)
                    del segments[idx]
                    segments.insert(
                        idx, b
                    )  # insert b first so that a remains at the same index
                    segments.insert(idx, a)
                    print(
                        f"-- New segments are {segment_duration(a)} and {segment_duration(b)} seconds long"
                    )
                    idx += 2  # skip the two newly added segments
                else:
                    idx += 1

            print(f"- Iteration num: {iterations}")
            iterations += 1
        iterations = 1

        min_count = len(
            [seg for seg in segments if segment_duration(seg) < min_duration]
        )
        print(f"Segment found based on sentence boundary: {total_segments_before}")
        print(f"After splitting long sentences there are {len(segments)} segments")
        print(f"Found {min_count} that where less then {min_duration}")

        print(f"Flattening the segments")
        segments = [flatten(seg) for seg in segments]

        # Let's combine segments until we reach the maximum duration
        idx = 0
        while idx < len(segments) - 1:  # check until the second last segment
            seg = segments[idx]
            next_seg = segments[idx + 1]
            combined_seg = merge_segments(seg, next_seg)

            if combined_seg["duration"] <= max_duration:
                segments[
                    idx
                ] = combined_seg  # replace the current segment with the merged one
                del segments[
                    idx + 1
                ]  # delete the next segment which is now part of the merged one
            else:
                idx += 1

        print(f"After combining there are {len(segments)} segments")
        print(
            f"Of which {len([seg for seg in segments if seg['duration'] < min_duration])} are less than {min_duration} seconds."
        )
        print(f"Remving segments that are less than {min_duration} seconds.")

        segments = [seg for seg in segments if seg["duration"] >= min_duration]
        print("Remvoing segments that only have <unk> or [hik: ...].")
        segments = [seg for seg in segments if seg["text_norm"].strip().rstrip()]

        out_folder = os.path.join(output_folder, fileId2split[file_id], file_id)
        os.makedirs(out_folder, exist_ok=True)
        print(f"Saving the segments to {out_folder}")
        for idx, segment in enumerate(segments):
            filename = f"{file_id}_{str(idx)}_{str(segment['duration'])}"

            out_f = os.path.join(out_folder, filename)

            extract_audio_segment(
                audio_file, out_f + ".wav", segment["start"], segment["duration"]
            )
            norm_text = out_f + "_norm.txt"
            text = out_f + ".txt"
            if not os.path.exists(norm_text):
                with open(norm_text, "w") as f:
                    f.write(segment["text_norm"])
            else:
                print(f"{norm_text} exists, wont overwrite")

            if not os.path.exists(text):
                with open(text, "w") as f:
                    f.write(segment["text"])
            else:
                print(f"{text} exists, wont overwrite")
            line = [
                file_id,
                filename,
                segment["text_norm"],
                segment["text"],
                str(segment["start"]),
                str(segment["end"]),
                str(segment["duration"]),
                out_f + ".wav",
            ]
            info.append(line)

    dev_trans = os.path.join(output_folder, "dev.trans")
    test_trans = os.path.join(output_folder, "test.trans")
    train_trans = os.path.join(output_folder, "train.trans")

    if any(
        [
            os.path.exists(dev_trans),
            os.path.exists(test_trans),
            os.path.exists(train_trans),
        ]
    ):
        print(f"{dev_trans}, {test_trans} or {train_trans} exist, wont overwrite.")
        return dev_trans, test_trans, train_trans

    with open(test_trans, "w") as test, open(dev_trans, "w") as dev, open(
        train_trans, "w"
    ) as train, open(test_trans.replace(".trans", ".info"), "w") as test_info, open(
        dev_trans.replace(".trans", ".info"), "w"
    ) as dev_info, open(
        train_trans.replace(".trans", ".info"), "w"
    ) as train_info:
        header = " ".join(["file_id", "segment_id", "start", "end", "duration"]) + "\n"
        test_info.write(header)
        dev_info.write(header)
        train_info.write(header)

        for line in info:
            file_id = line[0]
            info = "\t".join(line[:2] + line[4:-1]) + "\n"
            line = "\t".join([line[-1], line[2]]) + "\n"
            if fileId2split[file_id] == "test":
                test.write(line)
                test_info.write(info)
            elif fileId2split[file_id] == "dev":
                dev.write(line)
                dev_info.write(info)
            elif fileId2split[file_id] == "train":
                train.write(line)
                train_info.write(info)
            else:
                raise Exception("File not in splits")
    return dev_trans, test_trans, train_trans
