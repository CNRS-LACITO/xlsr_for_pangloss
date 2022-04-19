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
from collections import defaultdict


# ----------- Functions ----------- #
def get_files_from_directory(path):
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".wav")]
    return files


def predict_audio(arguments):
    # Preprocessing the data
    model = Wav2Vec2ForCTC.from_pretrained(arguments.model_dir).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(arguments.model_dir)
        
    # liste des fichiers du répertoire
    files = get_files_from_directory(arguments.input_dir)
        
    dico = defaultdict(list)
    
    # dictionnaire : clé = prefix (nom du fichier audio) et valeur = liste des chunk pour ce fichier
    for filename in files:
        prefix = "_".join(filename.split("_")[:-1])
        dico[prefix].append(filename)
    
    
    
    for prefix in dico:
        
        predictions = []

        for i in sorted(dico[prefix]):
            speech_array, sampling_rate = torchaudio.load(arguments.input_dir+i)
            
            speech = speech_array[0].numpy()
           
            input_dict = processor(speech, return_tensors="pt", padding=True, sampling_rate=16_000)
            
            with torch.no_grad():
                logits = model(input_dict.input_values.to("cuda")).logits
            
            pred_ids = torch.argmax(logits, dim=-1)[0]
            predictions.append(processor.decode(pred_ids))
          

        df_results = pd.DataFrame({'File': sorted(dico[prefix]),
                                   'Prediction': predictions})
        df_results.to_csv(arguments.input_dir + "/" + prefix+"_"+arguments.model_dir.split("/")[-2] + ".tsv", index=False, sep='\t')
    


# ----------- Arguments ----------- #
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

gen_audio = subparsers.add_parser("predict",
                                  help="Generate predictions from fine-tuned model for wav input files.")
gen_audio.add_argument('--input_dir', type=str, required=True,
                       help="Directory to wav files to transcribe.")
gen_audio.add_argument('--model_dir', type=str, required=True,
                       help="Directory where the fine-tuned model is stored.")

gen_audio.set_defaults(func=predict_audio)

arguments = parser.parse_args()
arguments.func(arguments)
