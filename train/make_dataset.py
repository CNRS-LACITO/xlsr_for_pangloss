import os
import xml.etree.ElementTree as et
from os import listdir
from os.path import join, isfile
import sox
from pathlib import Path
import csv
import pandas as pd
import argparse
from pydub import AudioSegment
import librosa
from sklearn.utils import shuffle
import sys
import soundfile as sf
from collections import defaultdict
import re

import hanzidentifier


MISC_SYMBOLS = { '~', '=', '¨', '↑', 'ː', '#', '$', 'X', '*', "+", "-", "_", '[', ']', '\ufeff'}
PUNC_SYMBOLS = {',', '!', '.', ';', '?', "'", '"', '“', '”', '…', '«', '»', ':', '«', '»', "ʔ", ' ̩'}

PINYIN = {}    


def get_files_from_directory(path):
    """
    Get all files from directory
    :param path: path where transcripts + wav files are stored
    :return: files
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files



def resample(args):
    path = args.path
    list_audio = get_files_from_directory(path+ 'wav/')
    for audio_file in list_audio:
        if audio_file.endswith('.wav'):
            signal, sampling_rate = librosa.load(path+ 'wav/' + audio_file)   
            signal_mono = librosa.to_mono(signal)
            signal = librosa.resample(signal, sampling_rate, 16_000)
            
            sf.write(path+ 'wav/' +'wav_resampled/'+audio_file, signal_mono, 16000, 'PCM_16')
     



def extract_information(xml_file):
    """
    Extract the transcription timecodes and ID from an xml file at the sentence level
    :param xml_file: transcription file
    :return: information (start audio, end audio, sentence id)
    """
    information = {}
    tree = et.parse(xml_file)
    root = tree.getroot()  # lexical resources
    sentences = root.findall('S')

    for child in sentences:
        id = child.attrib.get('id')
        transcript = child.find('FORM').text
        timecode = child.find('AUDIO').attrib
        info = [timecode['start'], timecode['end'], id]
        information[transcript] = info

    return information





def create_audio_tsv(args):
    """
    Create audios at the sentence level and create a tsv file which links each new audio file with the corresponding sentence
    :param args: path
    """
    files_process = []
    path = args.path
    files = get_files_from_directory(path + 'wav/')

    tsv = open(path + 'all_intact.tsv', 'wt')
    tsv_writer = csv.writer(tsv, delimiter='\t')
    tsv_writer.writerow(['path', 'sentence'])
    
    
    # tsv avec le pinyin recodé avec des caractères spéciaux mathématiques pour éviter la confusion entre les langues qui s'écrivent de la même manière
    tsv_modif = open(path + 'all.tsv', 'wt')
    tsv_writer_modif = csv.writer(tsv_modif, delimiter='\t')
    tsv_writer_modif.writerow(['path', 'sentence'])
    
    duree = 0
    nb_mots_pinyin = 0
    
    for f in reversed(sorted(files)):
    # for f in sorted(files):
        
        try:
            # traitement de certains fichiers japhug
            name_xml = 'crdo-JYA_' + f[:-4].upper() + '.xml'
            xml_file_path =Path(path + 'trans/' + name_xml)
            
            if xml_file_path.is_file():
                info = extract_information(path + 'trans/' + name_xml)
            else:
                name_xml = f[:-4].upper() + '.xml'
                info = extract_information(path + 'trans/' + name_xml)
                            
            wav_dir = Path(path) / "wav" / "clips/"
            wav_dir.mkdir(exist_ok=True, parents=True)

            wav_file = path + 'wav/' + f
            files_process.append(f)
            
            
            for k, v in info.items():
                for symb in MISC_SYMBOLS:
                    k=k.replace(symb, '')
                for punct in PUNC_SYMBOLS:
                    k=k.replace(punct, ' '+punct+' ')
                k = k.strip()
                k = " ".join(k.split())
                if k == '':
                    continue
                if float(v[1]) - float(v[0]) <= 1 or float(v[1]) - float(v[0]) > 20:
                    continue
                else:
                    tfm = sox.Transformer()
                    tfm.trim(float(v[0]), float(v[1]) + 0.2)
                    tfm.compand()
                    output_wav = f[:-4] + '_' + v[2] + '.wav'
                    tfm.build_file(wav_file, str(wav_dir) + '/' + output_wav)
                    
                    # comptage des phrases pinyin
                    if '@' in k:
                       nb_mots_pinyin += k.count('@')
                        
                    
                    if True:
                        
                        temp = k.split(' ')
                        texte = ''
                        
                        for mot in temp:
                            mot_modif = ''
                            caract_modif = ''
                            if not mot:
                                continue
                            if mot[0] == '@':
                                # nb_mots_pinyin +=1
                                # mot = mot[1:]
                                
                                for caract in mot:
                                    caract_modif = PINYIN.get(caract, caract)
                                    mot_modif += caract_modif
                                    continue
                                    
                                mot_modif = mot
                            
                            elif hanzidentifier.has_chinese(mot) :
                                for caract in mot:
                                    
                                    # suppression des caractères chinois
                                    if hanzidentifier.has_chinese(caract):
                                        caract_modif = ''
                                        mot_modif += caract_modif
                                    else:
                                        mot_modif += caract
                                
                            else:
                                mot_modif = mot  
                                
                            texte += mot_modif+' '
                            texte = texte.replace("/", "")
                            texte = re.sub("\(.*\)", "", texte)
    
                            

                        tsv_writer_modif.writerow([output_wav, texte])  
                    
                        # calcul de la duree totale de l'audio pris pour l'apprentissage 10800 = 3h 5400 = 1h30 3600 = 1h 21600=6h
                        duree += float(v[1]) - float(v[0])                   
                        if (duree >= 28800):
                        # if False:
                            print ('nb mots pinyin recontrés : ',nb_mots_pinyin)
                            message = str(duree/60)+"min préparées. Lancez maintenant le create_dataset !"
                            sys.exit(message)
                        
                        tsv_writer.writerow([output_wav, k])
        except Exception as e:
            
            print('Pb fichier: ', name_xml)
            print (e)
    print (duree)     
    tsv.close()


def create_dataset(args):
    """
    Create train/val/test tsv files (ratio 80 / 10 / 10)
    :param args: path
    """
    path = args.path

    corpus = pd.read_csv(path + 'all.tsv', sep='\t')
    corpus = shuffle(corpus)

    size_corpus = corpus.shape[0]

    split = [int(size_corpus * 0.8), int(size_corpus * 0.1)]

    train = corpus.iloc[:split[0]]
    val = corpus.iloc[split[0]:split[0] + split[1]]
    test = corpus.iloc[split[0] + split[1]:]

    train.to_csv(path + 'train.tsv', index=False, sep='\t')
    val.to_csv(path + 'valid.tsv', index=False, sep='\t')
    test.to_csv(path + 'test.tsv', index=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    create_audio = subparsers.add_parser("create_audio",
                                         help="Create audio per sentences from xml and wav and store the info in a tsv file. "
                                              "Make sure the transcriptions are in /trans and wav in /wav")
    create_audio.add_argument('--path', type=str, required=True,
                              help="path of the corpus with wav and transcription files.")
    create_audio.set_defaults(func=create_audio_tsv)

    split_dataset = subparsers.add_parser("create_dataset",
                                          help="Create dataset - train/val/test tsv files.")
    split_dataset.add_argument('--path', required=True, help="path of the corpus with wav and transcription files")
    split_dataset.set_defaults(func=create_dataset)

    
    resample_audio = subparsers.add_parser("resample_audio",
                                         help="Resample all audio files"
                                              "Make sure the audio files are in /wav")
    resample_audio.add_argument('--path', type=str, required=True,
                              help="path of the corpus to resample.")
    resample_audio.set_defaults(func=resample)

    

    args = parser.parse_args()
    args.func(args)
