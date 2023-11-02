########################################################################

# Author   : Carlos Daniel Hernández Mena and David Erik Mollberg
# Date     : August 05th, 2023
# Location : Reykjavík University

# Description:

# This is script calculates the Word Error Rate (WER) of the Test
# and Dev portions of the corpus "spjallromur_asr".
########################################################################


import jiwer


def jiwer_wer(reference, hypothesis):
    WER = jiwer.wer(reference, hypothesis)
    WER = WER * 100.0
    WER = round(WER, 3)
    return str(WER)


def jiwer_cer(reference, hypothesis):
    CER = jiwer.cer(reference, hypothesis)
    CER = CER * 100.0
    CER = round(CER, 3)
    return str(CER)


def calculate_wer(asr_results: str, results_file: str, split: str) -> None:
    hyp = [x.split("\t")[2].rstrip() for x in open(asr_results)]
    ref = [x.split("\t")[1].rstrip() for x in open(asr_results)]

    assert len(ref) == len(hyp), f"Transcripts and hypothesis are not equally long"
    wer = jiwer_wer(ref, hyp)

    res = f"wer ({split}): {wer}% [ {len(hyp)} hyp / {len(ref)} {asr_results}]"
    with open(results_file, "a") as f_out:
        f_out.write(res + "\n")

    print(res)


def calculate_cer(asr_results: str, results_file: str, split: str) -> None:
    hyp = [x.split("\t")[2].rstrip() for x in open(asr_results)]
    ref = [x.split("\t")[1].rstrip() for x in open(asr_results)]

    assert len(ref) == len(hyp), f"Transcripts and hypothesis are not equally long"
    cer = jiwer_cer(ref, hyp)

    res = f"cer ({split}): {cer}% [ {len(hyp)} hyp / {len(ref)} {asr_results}]"
    with open(results_file, "a") as f_out:
        f_out.write(res + "\n")

    print(res)
