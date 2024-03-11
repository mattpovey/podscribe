# Script to fetch all episodes of a podcast from the RSS feed
# and save them to a local directory. Each podcast episode is
# saved as a given the same name as its title. All titles, 
# descriptions, and URLs are saved to the metadata CSV file.

# THIS SCRIPT ONLY TOUCHES AUDIO FILES AND TRANSCRIPTS, NOT DATA

import feedparser
import os

# Import functions from libPodSemSearch
from libPodSemSearch import gen_filenames, \
    add_episodes, \
    record_metadata, \
    fetch_episodes, \
    check_dir, \
    convert_to_wav, \
    transcribe_episodes

# Import config from config.py
from config import feed_url, \
    pod_prefix, \
    pod_dir, \
    tscript_dir, \
    save_dir, \
    episode_dir, \
    wav_dir, \
    csv_dir, \
    transcription_dir, \
    metadata_csv, \
    tscript_mode, \
    whisper_model

# Set the name of the podcast - all directories and files will use this identifier
# pod_prefix="./rihPodcast"

# Set the directory paths
# pod_dir=f'{pod_prefix}_podcast'
# tscript_dir=f'{pod_prefix}_transcripts'
# save_dir = f'{pod_prefix}/audio'
# episode_dir = f'{pod_prefix}/audio'
# wav_dir = f'{pod_prefix}/wav'
# csv_dir = f'{pod_prefix}/csv'
# transcription_dir = f'{pod_prefix}/transcripts'

# Set the filenames
# metadata_csv = f'{csv_dir}/episodeMetadata.csv'  

# Create a list to store new episodes if the metadata file does not exist

# Check that all directories exist
# First that we have a directory for the podcast
check_dir(pod_prefix, count_files=0, create=1)
# Then the subdirectories
check_dir(pod_dir, count_files=0, create=1)
check_dir(episode_dir, count_files=1, create=1)
check_dir(wav_dir, count_files=0, create=1)
check_dir(save_dir, count_files=0, create=1)
check_dir(csv_dir, count_files=0, create=1)
check_dir(transcription_dir, count_files=0, create=1)

new_episodes = []    
    
# Podcast RSS feed URL
# feed_url = 'https://therestishistory.supportingcast.fm/content/eyJ0IjoicCIsImMiOiIxNDc3IiwidSI6Ijc3MTYzNiIsImQiOiIxNjM0OTQwODcyIiwiayI6MjY3fXw2MTViMDljYTBhNTYzNjcxZmI1ZTc0NjJiNmNkMDNmOTA4NjU0NWQ0MWJlOGY3NDgyZGVlNDRjMjVjNjA3ZDZi.rss'
try:
    rih_feed = feedparser.parse(feed_url)    # Parse RSS feed
except:
    print("Error parsing RSS feed")
    exit()
episode_dict = gen_filenames(rih_feed)    # Generate a dictionary of episode metadata including filenames

# Add the new episodes to the metadata dataframe
df_metadata = add_episodes(metadata_csv, episode_dict)
record_metadata(metadata_csv, df_metadata)

# Fetch the episodes and save to save_dir
# fetch_episodes checks if episodes are already downloaded
fetch_episodes(episode_dict, save_dir)

# Convert all audio files to wav
# Convert to wav checks if wav files already exist
convert_to_wav(episode_dir, wav_dir)

# for each fil#e in the wav_path directory check whether a corresponding
# transcript exists. Both filnames should be the same except for
# the extension. Wav files have the .wav extension and transcripts
# have the .srt extension. If a transcript does not exist, add the wav file
# to the list of files to be transcribed.

file_list = []
for wav_file in os.listdir(wav_dir):
    for srt_file in os.listdir(transcription_dir):
        w = os.path.splitext(wav_file)[0]
        s = os.path.splitext(srt_file)[0]
        if w == s:
            break
    else:
        file_list.append(wav_file)

for f in file_list:
    transcribe_episodes(wav_dir, transcription_dir, "srt", [f], tscript_mode, whisper_model)
