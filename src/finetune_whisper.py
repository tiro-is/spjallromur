import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, Dataset, IterableDatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
import os


def convert(model_dir: str) -> str:
    output = f"{model_dir}_ct2"
    for f in ["config.json", "model.safetensors"]:
        if not os.path.exists(os.path.join(model_dir + f"/{f}")):
            raise ValueError(
                f"File {f} does not exist in {model_dir}. Create a symlink from the best checkpoint."
            )

    command = [
        "ct2-transformers-converter",
        "--model",
        model_dir,
        "--output_dir",
        output,
        "--copy_files",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "--quantization",
        "float16",
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted the model: {output}")
    except subprocess.CalledProcessError as e:
        raise (f"An error occurred: {e}")
    return output


def load_data(file_path):
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            audio_path, transcript = line.rstrip().strip().split("\t")
            data.append({"audio": audio_path, "transcript": transcript})
    return data


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def finetune(
    whisper_model: str,
    dev_trans: str = "segmented/dev.trans",
    test_trans: str = "segmented/train.trans",
    train_trans: str = "segmented/test.trans",
    output_dir: str = "./whisper-large-icelandic-30k-steps-1000h-spjallromur-test",
):
    def prepare_dataset(batch):
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["transcript"]).input_ids
        return batch

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    train_data = load_data(
        train_trans,
    )
    dev_data = load_data(dev_trans)
    test_data = load_data(test_trans)

    # Create Hugging Face datasets
    spjallromur = IterableDatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "audio": [d["audio"] for d in train_data],
                    "transcript": [d["transcript"] for d in train_data],
                }
            ),
            "dev": Dataset.from_dict(
                {
                    "audio": [d["audio"] for d in dev_data],
                    "transcript": [d["transcript"] for d in dev_data],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "audio": [d["audio"] for d in test_data],
                    "transcript": [d["transcript"] for d in test_data],
                }
            ),
        }
    )

    spjallromur = spjallromur.cast_column("audio", Audio(sampling_rate=16000))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        whisper_model, language="Icelandic", task="transcribe"
    )
    spjallromur = spjallromur.map(prepare_dataset).with_format("torch")

    processor = WhisperProcessor.from_pretrained(
        whisper_model, language="Icelandic", task="transcribe"
    )
    metric = evaluate.load("wer")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=1000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=spjallromur["train"],
        eval_dataset=spjallromur["dev"],  # evaluation dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=None,
    )

    processor.save_pretrained(training_args.output_dir)
    trainer.train()
