# -*- coding: utf-8 -*-
# ----------- Libraries ----------- #
import argparse
from os import listdir
from os.path import isfile, join
import librosa
import torchaudio
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import numpy as np
from pathlib import Path

from transformers import AutoProcessor
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM


# ----------- Functions ----------- #
def get_files_from_directory(path):
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".wav")]
    return files


def predict_audio(arguments):

    # processor = AutoProcessor.from_pretrained(Path(arguments.model_dir), eos_token=None, bos_token=None)
    processor = Wav2Vec2Processor.from_pretrained(arguments.model_dir, eos_token=None, bos_token=None)
    
    vocab_dict = processor.tokenizer.get_vocab()
    print(vocab_dict)
    # sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    
    # print(sorted_vocab_dict)
    # print(sorted_vocab_dict["Z"])
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=arguments.lm_file,
    )
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )   
    
    print(processor.tokenizer.get_vocab())
    print(len(processor_with_lm.tokenizer.get_vocab()))
    print(decoder._idx2vocab)
    print([(ord(v),v)for v in decoder._idx2vocab.values()if len(v)==1])
    # Preprocessing the data
    model = Wav2Vec2ForCTC.from_pretrained(arguments.model_dir).to("cuda")
    # processor = Wav2Vec2Processor.from_pretrained(arguments.model_dir)
    model_name =  arguments.model_dir.split('/')[-2]
    predictions = []

    files = get_files_from_directory(arguments.input_dir)
    predict_text = ""
    
    for i in files:
        speech_array, sampling_rate = torchaudio.load(arguments.input_dir+i)
        speech = speech_array[0].numpy()
        speech = librosa.resample(np.asarray(speech), 44_100, 16_000)
        input_dict = processor(speech, return_tensors="pt", padding=True, sampling_rate=16000)
        # input_dict = processor_with_lm(speech, return_tensors="pt", padding=True, sampling_rate=16000)
        with torch.no_grad():
            print(model(input_dict.input_values.to("cuda")).logits.shape)
            logits = model(input_dict.input_values.to("cuda")).logits.cpu().numpy()[0]
            print(logits.shape)
            print("-----")
        
        predict_text += decoder.decode(logits).replace("|"," ")
        predictions.append(decoder.decode(logits).replace("|"," "))
        # pred_ids = torch.argmax(logits, dim=-1)[0]
        # predictions.append(processor.decode(pred_ids))
        # predict_text += " ("+i+") "+processor.decode(pred_ids)
    
    df_results = pd.DataFrame({'File': files,
                                   'Prediction': predictions})
    df_results.to_csv(arguments.input_dir +'/'+ model_name + '_results.csv', index=False, sep='\t')
    
    with open (arguments.input_dir + '/'+ model_name + '_result.txt', 'w') as file_out:
        file_out.write(predict_text)
        
    # print (predict_text)    

# ----------- Arguments ----------- #
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

gen_audio = subparsers.add_parser("predict",
                                  help="Generate predictions from fine-tuned model for wav input files.")
gen_audio.add_argument('--input_dir', type=str, required=True,
                       help="Directory to wav files to transcribe.")
gen_audio.add_argument('--model_dir', type=str, required=True,
                       help="Directory where the fine-tuned model is stored.")

gen_audio.add_argument('--lm_file', type=str, required=True,
                       help="file for language model.")
                       
gen_audio.set_defaults(func=predict_audio)

arguments = parser.parse_args()
arguments.func(arguments)
