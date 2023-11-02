########################################################################

# Author   : Carlos Daniel Hernández Mena & David Erik Mollberg
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.

# Description:

# This script transcribe the Dev and Test portions using the
# model of Faster-Whisper previously downloaded.

########################################################################

import re
import os
from tqdm import tqdm
from faster_whisper import WhisperModel
from multiprocessing import Process, Manager
from tqdm import tqdm


def transcribe_file(
    data_path: str,
    hyp_output: str,
    whisper_model: str,
    device: str = "cpu",
    compute_type: str = "int8",
) -> None:
    """
    Transcribes audio files using Faster-Whisper.

    Parameters:
    - data_path (str): Path to the input data file containing paths to audio files and their transcriptions.
    - hyp_output (str): Path to the output file where transcriptions will be written.
    - whisper_model (str): Path to the pretrained Faster-Whisper model.
    - device (str, optional): Device to which the model is sent. Defaults to 'cpu'.
    - compute_type (str, optional): Type of computation to be performed. Defaults to 'int8'.
    """

    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    audio_files = [x.split("\t") for x in open(data_path)]
    with open(hyp_output, "w") as f_out:
        for wav_file, transcript in tqdm(audio_files, total=len(audio_files)):
            wav_id = os.path.basename(wav_file).rstrip(".wav")
            hyp = ""
            segments, _ = model.transcribe(wav_file, beam_size=8)
            for segment in segments:
                hyp += segment.text + " "
            hyp = re.sub("\s+", " ", hyp).strip().rstrip()
            f_out.write(f"{wav_id}\t{transcript.rstrip()}\t{hyp}\n")


def transcribe_batch(
    sub_audio_files: list,
    whisper_model: str,
    results: list,
    device: str,
    compute_type: str,
) -> None:
    """
    Transcribes a batch of audio files in parallel.

    Parameters:
    - sub_audio_files (list): List of audio file paths and their transcriptions.
    - whisper_model (str): Path to the pretrained Faster-Whisper model.
    - results (list): List to collect results from the processes.
    - device (str): Device to which the model is sent.
    - compute_type (str): Type of computation to be performed.
    """

    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    for wav_file, transcript in sub_audio_files:
        wav_id = os.path.basename(wav_file).rstrip(".wav")
        hyp = ""
        segments, _ = model.transcribe(
            wav_file,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
        )
        for segment in segments:
            hyp += segment.text + " "
        hyp = re.sub("\s+", " ", hyp).strip().rstrip()
        results.append((wav_id, transcript.rstrip(), hyp))


def transcribe_file_parallel(
    data_path: str,
    hyp_output: str,
    whisper_model: str,
    device: str = "cpu",
    compute_type: str = "int8",
    batches: int = 5,
) -> None:
    """
    Transcribes audio files in parallel using Faster-Whisper.

    Parameters:
    - data_path (str): Path to the input data file containing paths to audio files and their transcriptions.
    - hyp_output (str): Path to the output file where transcriptions will be written.
    - whisper_model (str): Path to the pretrained Faster-Whisper model.
    - device (str, optional): Device to which the model is sent. Defaults to 'cpu'.
    - compute_type (str, optional): Type of computation to be performed. Defaults to 'int8'.
    - batches (int, optional): Number of parallel batches to be processed. Defaults to 5.
    """

    # Get the model
    audio_files = [x.split("\t") for x in open(data_path)]

    # Split the audio_files into chunks for parallel processing
    chunk_size = len(audio_files) // batches
    audio_file_chunks = [
        audio_files[i : i + chunk_size] for i in range(0, len(audio_files), chunk_size)
    ]

    # Use a Manager list to collect results from processes
    manager = Manager()
    results = manager.list()
    # Create and start processes

    print(f"Decoding {data_path} in {batches} batches")
    processes = []
    for i in range(batches):
        p = Process(
            target=transcribe_batch,
            args=(audio_file_chunks[i], whisper_model, results, device, compute_type),
        )
        processes.append(p)
        p.start()

    # Wait for processes to complete
    for p in processes:
        p.join()

    print(f"Writing the results to {hyp_output}")
    # Write the results to hyp_output
    with open(hyp_output, "w") as f_out:
        for wav_id, transcript, hyp in results:
            f_out.write(f"{wav_id}\t{transcript}\t{hyp}\n")
