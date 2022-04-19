# Import the AudioSegment class for processing audio and the
# split_on_silence function for separating out silent chunks.
from pydub import AudioSegment
from pydub.silence import split_on_silence
from argparse import ArgumentParser, RawTextHelpFormatter
from os import listdir
from os.path import join, isfile
import librosa
import soundfile as sf


# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def get_files_from_directory(path):
    """
    Get all files from directory
    :param path: path where transcripts + wav files are stored
    :return: files
    """
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".wav")]
    return files
 
 
def resample(args):
    path = args.path
    list_audio = get_files_from_directory(path)
    for audio_file in list_audio:
        if audio_file.endswith('.wav'):
            signal, sampling_rate = librosa.load(path+ '/' + audio_file)   
            signal = librosa.to_mono(signal)
            signal = librosa.resample(signal, sampling_rate, 16_000)
            
            
            sf.write(path+'wav_resampled/'+audio_file, signal, 16000, 'PCM_16')   
    return path+'wav_resampled/'

def cut_files(args, resampled_path):
    path = args.path
    files = get_files_from_directory(resampled_path)
    
    for f in files:
        
        # Load your audio.
        song = AudioSegment.from_wav(resampled_path+"/"+f)
    
        # Split track where the silence is 2 seconds or more and get chunks using
        # the imported function.
        chunks = split_on_silence(
            # Use the loaded audio.
            song,
            # Specify that a silent chunk must be at least 0.5 seconds or 500 ms long.
            min_silence_len=500,
            # Consider a chunk silent if it's quieter than -16 dBFS.
            # (You may want to adjust this parameter.)
            silence_thresh=-40
        )
    
        # Process each chunk with your parameters
        for i, chunk in enumerate(chunks,start = 1):
            # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
            silence_chunk = AudioSegment.silent(duration=500)
    
            # Add the padding chunk to beginning and end of the entire chunk.
            audio_chunk = silence_chunk + chunk + silence_chunk
    
            # Normalize the entire chunk.
            normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
    
            # Export the audio chunk with new bitrate.
            # pour les 000000 print("Exporting chunk{0:06d}.wav.".format(i))
            print(f[:-4]+"_{0:06d}.wav.".format(i))
            normalized_chunk.export(
                args.folder+"/"+f[:-4]+"_{0:06d}.wav".format(i),
                bitrate="192k",
                format="wav"
            )


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument("--path", required=True,
                        help="file")
    parser.add_argument('--folder', required=True,
                        help="Place to save the generated files")

    args = parser.parse_args()

    resampled_path = resample(args)
    cut_files(args, resampled_path)
    