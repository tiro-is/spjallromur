Spjallromur - Icelandic Conversational Speech

## About the Spjallrómur corpus

Spjallromur is an open-source conversational speech corpus for speech technology development. The corpus is 21 hours and 20 minutes long, with 54 total conversations, and 102 speakers. The data was collected over the period of one year (September 2020 - September 2021) by Reykjavík University.

This is a revised version of Spjallromur. The original version did not include accurate timestamps, it can be found on [Clarin](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/187). For this revision timestamps were added by automatically aligning the audio files to the transcript. There is now also a script that transforms the audio files and transcripts into conventional short-segment ASR training data with defined test, development, and training sets.

Spjallrómur was collected using a custom made online chatting platform called Spjall, which is Icelandic for chat. Each speaker used their own microphones (some picked up background noise like the neighboring speakers or other speakers) and devices. The audio from each microphone (speaker) was saved to a separate audio file, .WAVE. There are two speakers per conversation. The speaker set contains both native and non-native Icelandic speakers. All speakers are adults. Due to some network lag there is sometimes a small difference in the length of the two audio files within a conversation. As there were a limited number of participants, some speakers may be in more than one conversation. The dataset was primarily created for automatic speech recognition but due to the nature of the dataset, it can also be used for other speech technology fields such as speaker identification, speaker diarization, and conversational language modeling.

The transcripts where created by first transcribing the audio and then manually reviewed and fixed. Personally identifiable information has been redacted in the audio with a 400H beep and replaced with `<BLLEP>` in the transcript. Partial words are marked with [HIK: ..].

- The full conversations contain 18 hrs 20 mins of 46 full conversations, 92 speakers.
- The half conversations contain 2 hrs 42 mins with 7 conversations.
- Unaligned data has 1 hr and 16 mins with 3 recordings.

There were three recordings that we were unable to align. The unaligned data is contains one conversation between two speakers and a half conversations, the other half was moved from full conversations in the original to half conversations in this revision. The original transcripts are still included in the folder as the text data can be used.

We manually evaluated the alignment by reviewing ~300 segments. Details on the alignment are found in the file `evaluation_of_alignment.md`.

## The structure of the corpus

    . - readme.txt
    . - metadata.tsv
    . - data/
            . - half_conversations/
                    . - 2a139f9b/
                            . - a_2a139f9b_20-29_m.json
                            . - a_2a139f9b_20-29_m.wav
            . - full_conversations/
                    . - 0f2c315c/
                            . - a_0f2c315c_30-39_f.json
                            . - a_0f2c315c_30-39_f.wav
                            . - b_0f2c315c_30-39_f.json
                            . - b_0f2c315c_30-39_f.wav
            . - unaligned/
            . - splits/

The file names are structured like `<spk-id>_<unique-key>_<age>_<gender>.wav`. Each audio file is 16 bit, 16000 kHz, single channel WAVE.

# ASR

The reason that the data was aligned in this revision is so that it can be used for training and evaluating ASR systems on conversational data. The script `run_asr_recipe.py` converts the corpus into short segmented audio clips as well as splitting them into `train`, `dev` and `test` sets.

The sets are as follows:

train.info

- Number of segments 5701
- Duration 22.56 hours
- 21 female and 58 male and 2 other

test

- Number of segments 286
- Duration ~1 hours
- 5 female and 5 male

dev

- Number of segments 334
- Duration ~1 hours
- 4 female and 4 male

There are is no speaker overlap between the sets, meaning speakers in the test set are not in the training set same applies to the dev set. The recipe also has scripts for running an ASR experiment. The results of this experiment are in `results/asr/whisper-large-icelandic-30k-steps-1000h-ct2.txt`

# Diarization

For use in diarization we provide a script that converts the corpus to a diarization-friendly format. The script `run_diarization_recipe.py` combines the full transcript of the converstations into a combined file both a `JSON` file and `RTTM` file is created.

The recipe also provides steps to replicate a diraization experiment using the [pyannote](https://github.com/pyannote/pyannote-audio). The results of the experiment are in `results/diarize/diarization_results.md`

## Authors

Reykjavík University

- Carlos Daniel Hernández Mena
- Judy Y Fong
- Staffan Hedström
- Ólafur Helgi Jónsson
- Lára Margrét H. Hólmfriðardóttir
- Sunneva Þorsteinsdóttir
- Málfriður Anna Eiríksdóttir
- Eydís Huld Magnúsdóttir
- Ragnheiður Þórhallsdóttir
- Jon Gudnason - jg@ru.is

Tiro ehf.

- David Erik Mollberg
- Luke James O'Brien

## Acknowledgements

Special thanks to the other members of the Language and Voice Lab (https://lvl.ru.is), the student employees, Róbert Kjaran, and Magnús Teitsson.

This project was funded by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by Almannarómur, is funded by the Icelandic Ministry of Education, Science and Culture.

This project was funded in part by the the Icelandic Directorate of Labour's student summer job program in 2021.

## Citations

@misc{fong-spjallromur,
title = {Spjallromur - Icelandic Conversational Speech},
author = {Fong, Judy Y and Hedstr{\"o}m, Staffan and J{\'o}nsson, {\'O}lafur
Helgi and H{\'o}lmfri{\dh}ard{\'o}ttir, L{\'a}ra Margr{\'e}t H. and
{\TH}orsteinsd{\'o}ttir, Sunneva and Eir{\'{\i}}ksd{\'o}ttir, M{\'a}lfri{\dh}ur
Anna and Mollberg, David Erik and Magn{\'u}sd{\'o}ttir, Eyd{\'{\i}}s Huld and
{\TH}{\'o}rhallsd{\'o}ttir, Ragnhei{\dh}ur and Gudnason, Jon},
url = {},
note = {{CLARIN}-{IS}},
copyright = {Creative Commons - Attribution 4.0 International ({CC} {BY} 4.0)},
year = {2022} }

## License

This dataset is released under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. (https://creativecommons.org/licenses/by/4.0/)
