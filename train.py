from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, BitsAndBytesConfig
import evaluate
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainer
from datasets import load_dataset, DatasetDict, Audio
import os

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "Chinese"
TASK = "transcribe"
CHECKPOINT_FOLDER = "checkpoints"
SAVE_FOLDER = 'saved_model'
DATASET_PATH = 'hokkien_yt.hf'
SEED = 42
SPLIT_DATASET = 1/8

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE, task=TASK)


def fetch_dataset(dataset_path, split=None):
    dataset = load_dataset(dataset_path)

    # Shuffle dataset
    dataset['train'] = dataset['train'].shuffle(seed=SEED)
    dataset['test'] = dataset['test'].shuffle(seed=SEED)

    if split:
        new_train_size = len(dataset['train']) * split
        new_test_size = len(dataset['test']) * split

        # Take the first 1/8 of the dataset
        dataset['train'] = dataset['train'].select(range(int(new_train_size)))
        dataset['test'] = dataset['test'].select(range(int(new_train_size)))

    return dataset


def prepare_dataset(dataset):
    def process_audio_and_labels(batch):
        audio = batch["audio"]

        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    return dataset.map(
        process_audio_and_labels, remove_columns=dataset.column_names["train"], num_proc=1)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def load_latest_checkpoint(checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir)
             if "-" in f and f.split("-")[-1].isdigit()],
            key=lambda x: int(x.split("-")[-1])
        )
        print(checkpoints)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f'loading {latest_checkpoint}')
            return os.path.join(checkpoint_dir, latest_checkpoint)
    return None


def train(model, dataset, save=None):
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Post-processing on the model
    model = prepare_model_for_kbit_training(model)

    # Lora
    config = LoraConfig(r=32, lora_alpha=64, target_modules=[
                        "q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_FOLDER,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=25,
        save_steps=500,  # CHANGE THIS
        # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        remove_unused_columns=False,
        label_names=["labels"],  # same reason as above
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=model_checkpoint)

    if save:
        model.save_pretrained(save)


if __name__ == '__main__':
    dataset = fetch_dataset(DATASET_PATH, split=SPLIT_DATASET)
    dataset = prepare_dataset(dataset)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model_checkpoint = load_latest_checkpoint(CHECKPOINT_FOLDER)
    if model_checkpoint:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_checkpoint, quantization_config=quantization_config, device_map="auto")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME, quantization_config=quantization_config, device_map="auto")

    train(model, dataset, save=SAVE_FOLDER)
