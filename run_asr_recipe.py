########################################################################

# Author   : Carlos Daniel Hernández Mena & David Erik Mollberg
# Date     : November 1st, 2023
# Location : Reykjavík University and Tiro ehf.

# This is scripts runs all the scripts that are necessary to perform
# an ASR experiment using the corpus Spjallromur and the Speech
# Recognition system Whisper.
#
# Dependencies can be installed by running `pip install -r requirements.asr` in a
# virtual environment. The script also requires a system installation of sox,
# installable on Debian-based systems with `apt install sox`.

########################################################################

import os

SPLITS_FOLDER = "splits"
SEGMENTED = "segmented"

########################################################################
from src.segment import run_segmentation

print("(1 of 4) Creating Segmented Files")
dev_trans, test_trans, train_trans = run_segmentation(
    SEGMENTED, SPLITS_FOLDER, min_duration=2, max_duration=20
)

# ########################################################################
# Download the ASR model "language-and-voice-lab/whisper-large-icelandic-30k-steps-1000h-ct2"

from src.check_dependencies import check_dependencies
from src.download_asr_model import download_asr_model

print(
    "(2 of 4) Checking dependencies and downloading the ASR model from Hugging Face ..."
)
# check_dependencies()
# whisper_model = "openai/whisper-medium"  # The large model requires alot of GPU memmory to train, one might what to use the medium model instead.
whisper_model = download_asr_model()


# ########################################################################
# Finetune the model using the Spjallrómur segments
from src.finetune_whisper import finetune

dev_trans = "segmented/dev.trans"
test_trans = "segmented/test.trans"
train_trans = "segmented/train.trans"

output_dir = "./whisper-large-icelandic-30k-steps-1000h-spjallromur"
finetune(
    whisper_model=whisper_model,
    dev_trans=dev_trans,
    test_trans=test_trans,
    train_trans=train_trans,
    output_dir=output_dir,
)

# ########################################################################
# Convert model from Hugging Face Transformers to Faster-Whisper format
from src.finetune_whisper import convert

print("(4 of 5) Convert model from Hugging Face Transformers to Faster-Whisper format")
finetuned_model = convert(output_dir)

# ########################################################################
# # Transcribe the Dev and Test splits using Faster-Whisper
from src.transcribe import transcribe_file, transcribe_file_parallel

print("(4 of 5) Transcribing the Dev and Test splits using Faster-Whisper ...")

output_dir = f"results/asr/{finetuned_model}"
os.makedirs(output_dir, exist_ok=True)
hyp_test = os.path.join(output_dir, "test")
hyp_dev = os.path.join(output_dir, "dev")


# Possible device and compute combinations for the Whisper decoder
#    device="cuda", compute_type="float16"
#    device="cuda", compute_type="int8"
#    device="cpu", compute_type="int8"
transcribe_file(
    test_trans, hyp_test, finetuned_model, device="cuda", compute_type="float16"
)
transcribe_file(
    dev_trans, hyp_dev, finetuned_model, device="cpu", compute_type="float16"
)

# Use the following to decode in parallel.
# transcribe_file_parallel(test_trans, hyp_test, whisper_model, device="cuda", compute_type="float16", batches=5)
# transcribe_file_parallel(dev_trans, hyp_dev, whisper_model, device="cuda", compute_type="float16", batches=5)

# ########################################################################
# Calculate the WER and CER of Dev and Test splits using jiwer
from src.score import calculate_cer, calculate_wer

print("(5 of 5) Calculating WER and CER of the Dev and Test splits ...")
calculate_wer(hyp_test, "results/results.txt", split="test")
calculate_wer(hyp_dev, "results/results.txt", split="dev")

calculate_cer(hyp_test, "results/results.txt", split="test")
calculate_cer(hyp_dev, "results/results.txt", split="dev")


# ########################################################################
print("\n")
print("---------------------------------------------------------")
print("           Recipe Successfully Executed!!!")
print("---------------------------------------------------------")

# ########################################################################
