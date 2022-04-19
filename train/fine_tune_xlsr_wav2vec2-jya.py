import json
import random

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchaudio

import datasets


from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from audiomentations import Compose, AddGaussianNoise, Gain, PitchShift, TimeStretch, Shift

SAMPLING_RATE = 16_000


def extract_all_chars(batch):
    all_text = "|".join(batch["sentence"])
    all_text = all_text.replace(" ", "|")

    # Be careful:
    # - the function has to return a dictionary
    # - the set has to be converted to a list
    #
    # to respect HF datasets API
    return {"vocab": list(set(all_text))}


def prepare_dataset(batch):
    assert all(
        sampling_rate == SAMPLING_RATE for sampling_rate in batch["sampling_rate"]
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids

    return batch


def wav2array(batch):
    """
    `batch` is dictionnary with keys:
    - `path`: the path to a wav in the clips directory XXX should be modified to include the full path
    - `sentence`: the preprocessed transcription
    """
    speech_array, sampling_rate = torchaudio.load("/data/user/m/cmacaire/exp_severine/japhug_data/clips/" + batch["path"])
    assert (
        speech_array.shape[0] == 1
    ), f"{batch['path']} is stereo file --- only mono files can be considered"
    assert (
        sampling_rate == SAMPLING_RATE
    ), f"The sampling rate of your data must be {SAMPLING_RATE:,}. {batch['path']} has a sampling rate of {sampling_rate:,}"

    # speech_array has the shape of [channel, time]
    # we are only considering the first channel in the following
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


# ------------ Dataclass ------------ #
@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv", type=str, required=True, help="Train .tsv file.")
    parser.add_argument("--test_tsv", type=str, required=True, help="Test .tsv file.")
    parser.add_argument(
        "--output_dir",
        type=lambda x: Path(x),
        required=True,
        help="Output directory to store the fine-tuned model.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_dataset("csv", data_files=[args.train_tsv], delimiter="\t")[
        "train"
    ]
    test_data = load_dataset("csv", data_files=[args.test_tsv], delimiter="\t")["train"]

    vocab_train = train_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_data.column_names,
    )
    vocab_test = test_data.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=test_data.column_names,
    )

    # why considering the test set vocabulary!!!
    vocab = set(vocab_train["vocab"]) | set(vocab_test["vocab"])

    vocab = {v: k for k, v in enumerate(vocab)}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    with open(args.output_dir / "vocab.json", "w") as vocab_file:
        json.dump(vocab, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        args.output_dir / "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    # here the `sampling_rate` argument correspond to the sampling
    # rate of the data the feature extractor will process --- it will
    # just check that this value correspond to the sampling rate used
    # to train the model (i.e. it will not sample the data nor check
    # that the value passed as paramater is actually the sampling rate
    # of the data.
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    processor.save_pretrained(args.output_dir)

    # load the wav
    train_data = train_data.map(wav2array, remove_columns=train_data.column_names)
    test_data = test_data.map(wav2array, remove_columns=test_data.column_names)


   
    
    # extract features
    train_data = train_data.map(
        prepare_dataset,
        remove_columns=train_data.column_names,
        batch_size=8,
        num_proc=4,
        batched=True,
    )
    test_data = test_data.map(
        prepare_dataset,
        remove_columns=test_data.column_names,
        batch_size=8,
        num_proc=4,
        batched=True,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")


    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.2,
        # attention_dropout=0.1,
        activation_dropout=0.1,
        hidden_dropout=0.05,
        # hidden_dropout=0.1,
        final_dropout=0.1,
        feat_proj_dropout=0.05,
        # feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        # mask_time_prob=0.075,
        layerdrop=0.04,
        # layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_zero_infinity=True,
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=16,
        # per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        # gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=60,
        fp16=True,
        save_steps=100,
        eval_steps=50,
        logging_steps=50,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
