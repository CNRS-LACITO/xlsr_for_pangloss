for filename in *.wav 
do
  output_filename=${filename/.wav/}
    ffmpeg -i $filename -ar 16000 -af "pan=mono|c0=FL" -acodec pcm_s16le resampled_wav/${output_filename}.wav
done