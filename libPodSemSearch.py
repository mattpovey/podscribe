import chromadb
from chromadb.utils import embedding_functions
import os
import re
import csv
# import openai
import tiktoken
import pandas as pd
import numpy as np
import pysrt
# import scipy
from IPython.display import clear_output
# from openai import embeddings_utils
# from openai import datalib
# import openai
from getpass import getpass
from tqdm import tqdm
# import time
# import uuid
import sys

# Fetch to Transcript imports
import feedparser
import requests
import platform
from datetime import datetime
import subprocess
import logging
# import json

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logger = logging.getLogger('tscript_logger')
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Fetch to Transcript functions
# -----------------------------------------------------------------------------
def gen_filenames(feed):
    # Check whether the title has illegal file system name characters
    # and replace them with underscores. 
    if platform.system == 'Windows':
        illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    else:
        illegal_chars = ['/', '\\', '*', '"', '<', '>', '|']

    # Create a dictionary of episode metadata
    episode_dict = {}
    # Get a count of episodes from the rss feed
    episode_count = len(feed.entries)
    for episodes in feed.entries:
        # Get the title, description and URL of the podcast episode
        episode_title = episodes.title
        # Replace illegal characters in titles with underscores
        for char in illegal_chars:
            episode_title = episode_title.replace(char, '-')
        episode_description = episodes.description
        episode_url = episodes.enclosures[0].href

        # The date needs to be converted to YYYYMMDD
        episode_date = episodes.published
        date_object = datetime.strptime(episode_date, "%a, %d %b %Y %H:%M:%S %z")
        # Format the date object as YYYYMMDD
        episode_date = date_object.strftime("%Y%m%d")

        # Generate a four digit episode number for the episode begining 
        # with episode_count and counting back to 1
        episode_number = str(episode_count).zfill(4)
        episode_count -= 1
        
        # Add the episode metadata to the episode_dict
        episode_dict[episode_title] = {'title': episode_title, 'date': episode_date, 'number': episode_number, 'description': episode_description, 'url': episode_url}

    return episode_dict

def add_episodes(metadata_csv, episode_dict):
    new_episodes = []
    # Attempt to load the CSV file to check whether the episode has already been recorded
    if os.path.exists(metadata_csv):
        df_metadata = pd.read_csv(metadata_csv, dtype={'number': str})

    for episode in episode_dict:
        title = episode_dict[episode]['title']
        url = episode_dict[episode]['url']
        description = episode_dict[episode]['description']
        date = episode_dict[episode]['date']
        number = str(episode_dict[episode]['number'])
        f_ext = url.split('.')[-1]
        # Set a destination file name and check whether it already exists
        filename = date + '_' + number + '_' + title + "." + f_ext

        # Record episodes, checking whether they have already been recorded
        if 'df_metadata' in locals():
            if title in df_metadata['title'].values:
                # print("Episode already recorded: " + title)
                pass
            else:
                # Add the new episode to the df_metadata dataframe
                new_episodes.append([title, date, number, description, filename, url])
                # print("New episode added to existing data: " + title)
        else:
            # We are building the metadata file for the first time
            new_episodes.append([title, date, number, description, filename, url])
            # print("New episode added: " + title)

    # If df_metadata is empty, we are building the metadata file for the first time
    # Create it from the new_episodes list
    if 'df_metadata' not in locals():
        df_metadata = pd.DataFrame(new_episodes, columns=['title', 'date', 'number', 'description', 'filename', 'url'])
        df_metadata = df_metadata.astype({'number': 'str'})
    else:
        # Add the new episodes to the df_metadata dataframe
        df_new_eps = pd.DataFrame(new_episodes, columns=['title', 'date', 'number', 'description', 'filename', 'url'])
        # df_new_eps = df_new_eps.astype({'number': 'str'})
        df_metadata = pd.concat([df_metadata, df_new_eps], ignore_index=True)
    
    print(f'Found {len(new_episodes)} new episodes for download.')
    return df_metadata

# Save the metadata to a CSV file
def record_metadata(metadata_csv, df_metadata):
    try:
        with open(metadata_csv, 'x') as file:
            file.write('Hello')
    except FileExistsError:
        print("Metadata File already exists.")

    try:
        with open(metadata_csv, 'w') as csvfile:
            # Sort the contents of df_metadata by episode number
            df_metadata.sort_values(by=['number'], inplace=True)
            # Write the contents of df_metadata to the CSV file
            df_metadata.to_csv(csvfile, header=True, index=False)
    except:
        print("Error writing metadata to file")
        exit()

def fetch_episodes(episode_dict, save_dir):
    # Fetch the episodes and save to save_dir. Use the episode_dict to
    # determine metadata and url since gen_filenames() has done the hard work
    for episode in episode_dict:
        title = episode_dict[episode]['title']
        url = episode_dict[episode]['url']
        description = episode_dict[episode]['description']
        date = episode_dict[episode]['date']
        number = episode_dict[episode]['number']
        f_ext = url.split('.')[-1]
        # Set a destination file name and check whether it already exists
        filename = save_dir + "/" + date + '_' + number + '_' + title + "." + f_ext
        # Don't download if it already exists
        if os.path.exists(filename):
            # print("Found ", filename, ".")
            # Also check whether the transcript already exists
            transcript = save_dir + "/" + date + '_' + number + '_' + title + ".srt"
            if os.path.exists(transcript):
                # print("Found ", transcript, ".")
                continue
            continue
        # Download the audio file
        # if int(number[1:]) > 400:
        print("Downloading ", episode, "...")
        response = requests.get(url)

        # Save the audio file in save_dir
        print("Saving ", title, " to ", save_dir, " as ", filename, "...")
        with open(filename, 'wb') as f:
            f.write(response.content)

# Check that directory exists and optionally print count of files
# Attempt to create the directory if it does not exist
# TODO: Use this more generally in the code
def check_dir(directory, count_files=0, create=0):
    # print("Checking for directory, " + directory + "with count_files = " + str(count_files) + " and create = " + str(create) + ".")
    if os.path.exists(directory):
        if count_files == 1:
            print("Found ", len(os.listdir(directory)), "files in", directory, ".")
            return True
        else:
            # print("Found ", directory, ".")
            return True
    else:
        print("Directory ", directory, "not found.")
        if create==1:
            print("Creating directory: ", directory, ".")
            try:
                os.makedirs(directory)
                print("Directory created.")
                return True
            except:
                print("Error creating directory: ", directory, ".")
                sys.exit()

# Convert each episode to a 16bit wav file
def convert_to_wav(episodes_dir, wav_dir):
    # Get list of files in episode_dir
    files = os.listdir(episodes_dir)
    for file in files:
        # Get the title of the episode
        title = os.path.splitext(file)[0]
        wav_path = wav_dir + "/" + title + '.wav'
        # Get the path to the episode
        episode_path = os.path.join(episodes_dir, file)
        if os.path.exists(wav_path):
            # print("Found ", wav_path, ".")
            continue
        # else:
        #     print("Converting ", file, " to 16bit wav file.")
        # Strip the extension from the episode file name and store in title
        title = os.path.splitext(file)[0]
        # Convert the episode to a 16bit wav file
        # print(f'Converting, {file} to 16bit wav file.')
        # print(f'Saving in, {wav_path}.')
        subprocess.run(["ffmpeg", "-hide_banner", "-i", episode_path, "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000",  wav_path])

# Transcribe each episode in the rih_wav directory
def transcribe_episodes(wav_dir, transcription_dir, out_format, file_list, tscript_mode, whisper_model):
    for file in file_list:
        # Check whether the transcription already exists and continue if so continue
        out_file = transcription_dir + "/" + os.path.splitext(file)[0] + '.' + out_format
        if os.path.exists(out_file):
            print("Found ", out_file)
            continue
        # Get the title of the episode
        title = os.path.splitext(file)[0]
        of_arg = '-o' + out_format
        # Get the path to the 16bit wav file
        episode_path = os.path.join(wav_dir, file)
        # Set the path of the transcription file. Extension is added by whisper.cpp.
        transcription_tdrz = os.path.join(transcription_dir,  "tdrz/", title)
        transcription_file = os.path.join(transcription_dir, title)

        # Check that the whisper.cpp/main executable exists
        if not os.path.exists('whisper.cpp/main'):
            print("whisper.cpp/main does not exist. Install whisper.cpp from https://github.com/ggerganov/whisper.cpp.")
            exit()
        # Run whisper.cpp
        if tscript_mode == "diarized" or tscript_mode == "both":
            print("Transcribing with diarization ", file, " to ", transcription_file, ".")
            subprocess.run(['whisper.cpp/main', "-tdrz", "-m", "whisper.cpp/models/ggml-small.en-tdrz.bin", of_arg, "-oj", "-otxt",  "-f", episode_path, "-of", transcription_tdrz])
        if tscript_mode == "single" or tscript_mode == "both":
            print("Transcribing in high quality ", file, " to ", transcription_file, ".")
            subprocess.run(['whisper.cpp/main', "-m", "whisper.cpp/models/ggml-medium.bin", of_arg, "-otxt", "-f", episode_path, "-of", transcription_file])


# -----------------------------------------------------------------------------
# PodtoVector functions
# -----------------------------------------------------------------------------
def save_csv(df, out_csv):
    out_csv = os.path.join(f'{out_csv}')
    save = input(f'Press Y/y to save the dataframe to {out_csv}')
    if save.lower() == 'y':            
        # Check if the file already exists and confirm overwrite
        if os.path.exists(out_csv):
            print(f'{out_csv} already exists')
            overwrite = input('Overwrite? (y/n): ')
            if overwrite.lower() == 'y':
                df.to_csv(out_csv, index=False)
                print("File saved")
            else:
                print("File not saved")
        else:
            print("Creating file")
            df.to_csv(out_csv, index=False)
    else:
        print("File not saved")

# Function to count tokens in a text
def tok_count(text, embeds_model):
    if embeds_model == "text-embeddings-ada-002":
        encoding = tiktoken.encoding_for_model(embeds_model)
        n_toks = len(encoding.encode(text))
    elif embeds_model == "all-MiniLM-L6-v2":
        # Count the words in text
        n_toks = len(text.split())  
    return n_toks

def nan_check(df):
    # Check for NANs across the dataframe. Strong indicator of problems.
    for col in df.columns:
        num_nans = df[col].isnull().sum()
        if num_nans > 0:
            print(f'*** {col} has {num_nans} NANs')
        else:
            print(f'{col} has no NANs')
    
# Delete an episode from the data for testing purposes
# def test_embed_setup():
#     df_embeds = pd.read_csv(embeds_csv)
#     # delete all rows of df_embeds where the filename is 20211127_0038_Communism.srt
#     df_embeds = df_embeds[df_embeds['filename'] != '20211127_0038_Communism.srt']
#     # Write out df_embeds to embeds_csvb
#     df_embeds.to_csv(embeds_csv, index=False)

# Work out what it's going to cost to embed the dataset
# TODO: Only count the cost of text that does not currently have a value for embedding
def check_raw(df_raw, tscript_dir, embeds_model, embeds_model_cost):
    tok_sum = df_raw['token_count'].sum()
    embeds_cost = (tok_sum / 1000) * embeds_model_cost
    tok_desc = df_raw['token_count'].describe()
    print(f'Whisper creates lines of different lengths in different runs of the same file')
    print(f'The token_count field is the number of tokens in the text field: \n{tok_desc} \n')
    print(f'The total number of tokens in the dataset is {tok_sum}')
    print(f'The cost of embedding the dataset using {embeds_model} at the current cost of {embeds_model_cost} per 1K tokens is USD ${embeds_cost:.2f}')

    # count the number of unique filenames in the dataframe
    u_f = df_raw['filename'].unique()
    print(f'The total number of files in the dataset is {len(u_f)}')
    # Get a list of the filenames in transcripts
    filenames = os.listdir(tscript_dir)
    raw_filenames = df_raw['filename'].unique()

    # Compare the two lists
    missing_files = [x for x in filenames if x not in raw_filenames]
    print(f'There are {len(missing_files)} files in transcripts that are not in the raw csv')

    # Sort u_f by the four digit number surrounded by underscores in the filename
    u_f = sorted(u_f, key=lambda x: int(x.split('_')[1]))

    # Find any duplicated numbers in the series
    duplicated = [x for x in u_f if u_f.count(x) > 1]

    print(f'There are {len(duplicated)} duplicated files in the data')

# Major functions
def process_srt_files(directory, csv_file, embeds_model):
    # Create the set of existing files. This is only populated if csv_file exists and has content
    existing_files = set()
    df_raw = pd.DataFrame()
    new_rows = []
    csv_header = ['filename', 'srt_index', 'tc_start', 'tc_end', 'text', 'token_count', 'embedding']

    # Check if the csv file exists. If so, read in the existing files so we don't process them again
    # If it does not exist, create it and write the header
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        # Populate the existing files set.
        existing_files = set(df_existing['filename'].unique())
    else:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    for filename in os.listdir(directory):
        # Only process .srt files that have not already been processed
        if filename.endswith('.srt') and filename not in existing_files:
            # print(filename)
            path = os.path.join(directory, filename)
            # Create a srt object for the file using pysrt
            # this allows the index, start and end times, and text of each subtitle to be accessed
            f_srt = pysrt.open(path)

            for i, sub in enumerate(f_srt):
                text = sub.text.strip()
                text = text.replace('\n', ' ')
                text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
                if not text:
                    continue
                tokens = tok_count(text, embeds_model)
                embedding = np.nan
                new_rows.append([filename, i+1, str(sub.start), str(sub.end), text, tokens, embedding])
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    df_raw = pd.read_csv(csv_file)
    return df_raw

def split_sentences(text):
    # TODO: REPLACE WITH LANGCHAIN OR ALTERNATIVE
    # Define the regular expression pattern to split sentences
    #sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\.)\s'
    sentence_pattern = r'(?<!\w\.\w)(?<=\.|\?|!)\s+(?=[A-Z])'

    # Split the text into sentences using the regular expression
    sentences = re.split(sentence_pattern, text)

    return sentences

def clear_stdout():
    # Use ANSI escape code to clear the screen and move cursor to the beginning
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()

def combine_text(raw_csv, chunks_csv, max_tokens, embeds_model):
    if 'df_raw' not in locals():
        if os.path.exists(raw_csv):
            print(f'Loading df_raw from {raw_csv}')
            df_raw = pd.read_csv(raw_csv)
        else:
            print(f"{raw_csv} does not exist.")
            exit()

    if os.path.exists(chunks_csv):
        print(f'Loading df_chunks from {chunks_csv}')
        df_chunks_existing = pd.read_csv(chunks_csv)
        df_chunks_existing_set = set(df_chunks_existing['filename'].unique())
    else:
        df_chunks_existing = pd.DataFrame()
        df_chunks_existing_set = set()

    new_df_list = []  # create a list to hold the new rows
    new_df = pd.DataFrame() # A new dataframe to hold the combined text.

    # Remove rows from df_raw for which we already have embeddings
    df_raw = df_raw[~df_raw['filename'].isin(df_chunks_existing_set)]

    file_names = df_raw['filename'].unique()
    print(f'Number of files in existing chunks_csv: {len(df_chunks_existing_set)}')
    print(f'Number of files to process: {len(file_names)}')
    new_files = [x for x in file_names if x not in df_chunks_existing_set]

    # Process rows for each filename
    n_files = 0
    for file in file_names:
        sub_df = df_raw[df_raw['filename'] == file]  # get rows for a specific filename
        i = 0
        while i < len(sub_df):
            text = sub_df.iloc[i, sub_df.columns.get_loc('text')]
            new_end = sub_df.iloc[i, sub_df.columns.get_loc('tc_end')]
            token_count = sub_df.iloc[i, sub_df.columns.get_loc('token_count')]
            # first check if this line is greater than max_tokens in length
            # If so, we try to split it into sentences. If we can't we truncate it.
            if token_count > max_tokens:
                sentences = split_sentences(text)
                if len(sentences) > 1:
                    # If there is more than one sentence, split the text into sentences and process each sentence
                    for sentence in sentences:
                        token_count = tok_count(sentence, embeds_model)
                        if token_count <= max_tokens:
                            # Create a new row and append it to the list
                            new_row = sub_df.iloc[i].copy()
                            new_row['text'] = sentence
                            new_row['token_count'] = token_count
                            new_df_list.append(new_row)
                        else:
                            # truncate the sentence to max_tokens and add it to the list
                            sentence = sentence[:max_tokens]
                            token_count = tok_count(sentence, embeds_model)
                            new_row = sub_df.iloc[i].copy()
                            new_row['text'] = sentence
                            new_row['token_count'] = token_count
                            new_df_list.append(new_row)
            else:
                # If the line is less than max_tokens in length, check if the next line can be added to it
                j = i  # use another variable to move forward while combining texts
                # Combine text and recalculate token count if condition is met
                while j < len(sub_df) - 1 and token_count + sub_df.iloc[j+1, sub_df.columns.get_loc('token_count')] <= max_tokens:
                    text += ' ' + sub_df.iloc[j+1, sub_df.columns.get_loc('text')]  # Adding a whitespace at the end of each row
                    token_count = tok_count(text, embeds_model)
                    # Update the new end-timecode from tc_end
                    new_end = sub_df.iloc[j+1, sub_df.columns.get_loc('tc_end')]
                    j += 1
                # If we are at the last line of a file, add it to the text regardless of token count
                if j == len(sub_df) - 1:
                    text += ' ' + sub_df.iloc[j, sub_df.columns.get_loc('text')]
                    new_end = sub_df.iloc[j, sub_df.columns.get_loc('tc_end')]
                    token_count = tok_count(text, embeds_model)
                # Create a new row and append it to the list
                new_row = sub_df.iloc[i].copy()
                new_row['text'] = text
                new_row['token_count'] = token_count
                new_row['tc_end'] = new_end
                new_df_list.append(new_row)
                i = j + 1  # update i to move to the next unprocessed row in df
        # Clear the screen before updating progress
        # Check if I'm in a Jupyter notebook
        try:
            # Try to get the IPython instance
            ipython = get_ipython()
            if ipython is not None:
                # Clear the output cell
                clear_output(wait=False)
            else:
                clear_stdout()
        except NameError:
            clear_stdout()

        n_files += 1
        print(file)
        print(f"Episodes processed: {n_files}. Rows in new_df: {len(new_df_list)}")

    if len(file_names) == 0:
        print('No new files to process. Returning')
        new_df = df_chunks_existing.copy()
        return new_df, new_files
    else:
        # Create a new DataFrame from the list of rows
        new_df = pd.concat(new_df_list, axis=1).transpose()

        # Concatenate the new_df with the existing df_chunks
        new_df = pd.concat([new_df, df_chunks_existing], axis=0)

        # Reset the index of the new DataFrame
        new_df.reset_index(drop=True, inplace=True)
        # Remove all newlines from the text as per
        new_df['text'] = new_df['text'].str.replace('\n', ' ')

        # Add an embedding columng to the df and fill with NAN
        new_df['embedding'] = np.nan
        # print(new_df.head())

    return new_df, new_files
    
# Add metadata to the chunks csv
def generate_link(line, url):
    # get the timecode from the line
    pattern = r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
    matches = re.search(pattern, line)
    # get the link to the episode
    if matches:
        start_time = matches.group(1)
        hours, minutes, seconds_milliseconds = start_time.split(':')
        seconds, milliseconds = seconds_milliseconds.split(',')
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
        # Round total_seconds to the nearest second
        total_seconds = round(total_seconds) - 15
        link = url + "#t=" + str(total_seconds) + ".0"
    else:
        print("regex failed")
        link = url
    return link

def add_metadata(input_csv, metadata_csv, embeds_model, feed_url):
    # Need to grab the url from the feed because I'm not doing this properly...
    if not os.path.exists(metadata_csv):
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

    df_updates = pd.read_csv(input_csv)
    df_meta = pd.read_csv(metadata_csv)
    print(df_meta.head())

    # Add a url column to the df_updates dataframe after the filename column
    df_updates['title'] = pd.NA
    df_updates['number'] = pd.NA
    df_updates['date'] = pd.NA
    df_updates['url'] = pd.NA

    # Set the correct column order
    col_ord = ['title', 'number', 'date', 'filename', 'srt_index', 'tc_start', 'tc_end', 'url', 'text', 'embedding']

    # Recreate the dataframe with the correct column order
    df_updates = df_updates[col_ord]
    df_updates.head()

    # Add a column to the dataframe with a link to the episode timecode 
    for i, row in df_updates.iterrows():
        # Get the URL for the episode from the metadata by matching on the filename
        filename = row['filename']
        # Extract the episode number from the filename. It is three digits with a leading and following underscore
        pattern = r"_(\d{4})_"
        matches = re.search(pattern, filename)
        if matches:
            episode_number = matches.group(1)
        
        # Get the title, date, and number from the metadata
        title = df_meta[df_meta['filename'].str.contains(episode_number)]['title'].values[0]
        date = df_meta[df_meta['filename'].str.contains(episode_number)]['date'].values[0]
        number = df_meta[df_meta['filename'].str.contains(episode_number)]['number'].values[0]
        # Find the filename in the metadata which contains the episode number
        url = df_meta[df_meta['filename'].str.contains(episode_number)]['url'].values[0]

        # Format the timecode and text into a line
        timecode = row['tc_start'] + " --> " + row['tc_end']
        line = timecode + " " + row['text']
        link = generate_link(line, url)

        # Set the values in the dataframe
        df_updates.at[i, 'title'] = title
        df_updates.at[i, 'date'] = str(date)
        df_updates.at[i, 'number'] = str(number)
        df_updates.at[i, 'url'] = link
        df_updates.at[i, 'tokens'] = tok_count(row['text'], embeds_model)

    return df_updates



# -----------------------------------------------------------------------------
# ChromeDB functions (setup, query)
# -----------------------------------------------------------------------------
def setup_chromadb(dbname, collection_name, collection_metadata):
    chr_client = chromadb.PersistentClient(path=dbname)
    # TODO Move metadata to variable?
    try:
        collection = chr_client.get_collection(collection_name)
        print("Collection exists")
        print(f'There are {collection.count()} documents in {collection_name}')
    except:
        print("Collection does not exist")
        collection = chr_client.create_collection(name=collection_name, metadata=collection_metadata)
        print(f'There are {collection.count()} documents in {collection_name}')

    return chr_client, collection

# Compare the contents of the ChromeDB Collection with the contents of df_chunks
# Prepare a dataframe of files to add to the collection that are not already in the collection
def incremendal_add(df, chromadb_name, chroma_collection, collection_metadata, chunks_csv):    
    # Figure out which lines need to be added to the database
    # First get the list of filenames in the database
    chr_client, collection = setup_chromadb(chromadb_name, chroma_collection, collection_metadata)
    rih_col_mds = collection.get(
        include=["metadatas"]
    )

    files_in_coll = []
    for i in rih_col_mds['metadatas']:
        if i['filename'] not in files_in_coll:
            files_in_coll.append(i['filename'])

    print(f'Number of files in the collection: {len(files_in_coll)}')
    # print(f'files_in_coll: {files_in_coll}')

    # Compare the list of files in the collection with the list of files in the chunks csv
    # Remove any files from the chunks csv that are already in the collection
    if 'df_chunks' not in locals():
        print(f'Loading {chunks_csv} into df_chunks')
        df_chunks = pd.read_csv(chunks_csv)
    else:
        print(f'df_chunks already exists')

    # Compare the set of files in the collection with the set of files in the chunks csv
    files_in_csv = df_chunks['filename'].unique().tolist()
    files_to_add = set(files_in_csv) - set(files_in_coll)
    print(f'Number of files in the chunks csv: {len(files_in_csv)}')
    print(f'Number of files to add: {len(files_to_add)}')

    if len(files_to_add) == 0:
        print('No files to add. Exiting')
        return None
    else:
        # print(f'Files to add: {files_to_add}')
        # Create a dataframe of the files to add
        df_add = df_chunks[df_chunks['filename'].isin(files_to_add)]
        # reset the index of the dataframe
        df_add.reset_index(drop=True, inplace=True)

    return df_add


def query_chromadb(collection, query, num_results): 
    results = collection.query(
        query_texts=[query],
        n_results=num_results,
        include = ["documents", "metadatas", "distances"]
    )
    # print(results.keys())
    # delete the 'embeddings' key from the results dict
    # The 'include' statement stops the embeddings values being returned but the key is still there with NONE as the value' 
    results.pop('embeddings', None)
    results.pop('uris', None)
    results.pop('data', None)

    if len(results['documents']) == 0:
        print('No results found')
        return None
    else:
        try:
            df_ord = pd.DataFrame()
        except ValueError as e:
            print(f'Error: {e}')

        # Create columns for ids, distances, documents and metadata
        for key, value in results.items():
            # print("key = ", key, type(value))
            df_ord[key] = value[0]

        # Show the contents of the metadatas key
        # print(df_ord['metadatas'][0])


        # Sort the results by distance
        df_ord = df_ord.sort_values(by='distances', ascending=False)
        df_ord.reset_index(drop=True, inplace=True)
        # print(df_ord.head())
        res_eps = {}
        for i in range(num_results):
            ep_id = df_ord['ids'][i] #0
            ep_title = df_ord['metadatas'][i]['title'] #1
            ep_number = df_ord['metadatas'][i]['number'] #2
            ep_date = df_ord['metadatas'][i]['date'] #3
            ep_text = df_ord['documents'][i] #4
            ep_url = df_ord['metadatas'][i]['url'] #5
            ep_dist = df_ord['distances'][i] #6

            res_eps[f'result_{i}'] = [ep_id, ep_number, ep_dist, ep_title, ep_date, ep_text, ep_url] 

        # for i in res_eps:
        #     print(f'Result episode number: {res_eps[i][1]}')
        #     print(f'Similarity: {res_eps[i][2]}')
        #     print(f'Title: {res_eps[i][3]}')
        #     print(f'Released on: {res_eps[i][4]}')
        #     print(f'Quote: {res_eps[i][5]}')
        #     print(f'Listen here: {res_eps[i][6]}')
        #     print('\n')

    return res_eps

# -----------------------------------------------------------------------------
# GPT Functions
# -----------------------------------------------------------------------------
def check_key(model):
    # Test whether the API key is valid
    if "gpt" in model:
        import openai
        if openai.api_key == None:
            print('OpenAI API key is not set')
            openai.api_key = getpass(prompt="Enter your OpenAI API key: ")
            
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Please say hello to me!"},
                ]
            )
            print("API key is valid!")
            return True
        except openai.error.AuthenticationError:
            print("API key is invalid.")
            print('Please check your API key and try again')
            check_key(model)

    elif "mistral" in model:
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        print("setting mistral api key")
        if os.getenv("MISTRAL_API_KEY") != None:
            print("MISTRAL_API_KEY is set in environment variables")
            print(os.environ["MISTRAL_API_KEY"])
            if os.environ["MISTRAL_API_KEY"] == None:
                print("MISTRAL_API_KEY exists but is not set")
            else:
                print("MISTRAL_API_KEY is set in environment variables")
                mistral_key = os.environ["MISTRAL_API_KEY"]
        else: 
            mistral_key = getpass(prompt="Enter your Mistral API key: ")

        print("Creating Mistral Client with API key: ", mistral_key)
        mist_client = MistralClient(api_key=mistral_key)
        print("Attempting a test chat completion with Mistral AI.")
        try:
            response = mist_client.chat(
                model="mistral-tiny",
                messages=[
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content="Please say hello to me!"),
                ]
            )
            print(response.choices[0].message.content)
            print("API key is valid!")
        except:
            print("API key is invalid.")
            print('Please check your API key and try again')
            check_key(model)
    return mist_client

        
# Chunk text to fit inside token window
# If I ever get diarization working, this will need to be updated to handle speaker changes
def split_text(text, max_tokens):
    words = re.split(r'\s+', text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    i = 0
    while i < len(words):
        word = words[i]
        word_token_count = tok_count(word)
        current_token_count += word_token_count

        if current_token_count > max_tokens:
            # Find the nearest full stop after the token limit
            while "." not in word and i < len(words) - 1:
                i += 1
                word = words[i]
                current_token_count += tok_count(word)

            # Split at the full stop
            before_full_stop, after_full_stop = word.split(".", 1)
            current_chunk.append(before_full_stop + ".")
            chunks.append(" ".join(current_chunk))

            # Start a new chunk with the remaining part of the split word
            current_chunk = [after_full_stop.lstrip()]
            current_token_count = tok_count(after_full_stop)
        else:
            current_chunk.append(word)
        i += 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def gpt_proc(sys_role, user_role, gpt_model):
    # Relies on check_key() to have been run to check the API key
    # Create the system role which explains that GPT will answer questions about a podcast
    # In the rest of the role message, supply the user question and ask the model to provide 3 questions to send as searches.
    # sys_role_content_answer = f'You are an assistant which answers questions about a popular history podcast asked by fans of the podcast. You have been asked, \"{question}\". The user will provide some examples of episodes of the show where this subject was discussed and snippets from them. Please answer the fan\'s question based on this.'
    role_msgs = [] # List to store the role messages

    # Generate queries for semantic search

    if "gpt" in gpt_model:
        import openai
        print("OpenAI model")

        if gpt_model == "gpt-4" or gpt_model == "gpt-4-32k":
            role_msgs.append({"role": "system", "content": sys_role})
            # Add the current chunk to the role_msgs list
            role_msgs.append({"role": "user", "content": user_role})
        elif gpt_model != "gpt-4" and gpt_model != "gpt-4-32k":
            # Add the current chunk to the role_msgs list
            role_msgs.append({"role": "user", "content": user_role})
            # Add the system role content to the role_msgs list
            role_msgs.append({"role": "system", "content": sys_role})
        # Log the role messages
        # print("Role messages: ", role_msgs)
        logger.info (f"Role messages: {role_msgs} will be sent to {gpt_model}.")

        try:
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=role_msgs,
            )
        except openai.error.APIError as e:
            print(f"An API error occurred: {e}")
            logger.info (f"An API error occurred: {e}")
            raise e
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            logger.info (f"Failed to connect to OpenAI API: {e}")
            raise e
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            logger.info (f"OpenAI Rate limit exceeded: {e}")
            raise e
        except openai.error.AuthenticationError as e:
            print(f"Authentication error: {e} \n")
            print("Check your OpenAI API key is provided and correct. \n")
            logger.info (f"Authentication error: {e} \n")
            # jsonification is performed by the caller.
            raise e
            
        print("Response received.")
        r_text = response['choices'][0]['message']['content']
        return r_text
    
    elif "mistral" in gpt_model:
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        print("Mistral model")
        mist_client = check_key(gpt_model)
        # print(mist_client)
        mist_sysrole = ChatMessage(role="system", content=sys_role)
        mist_userrole = ChatMessage(role="user", content=user_role)
        role_msgs = [mist_sysrole, mist_userrole]

        # print("messages=", role_msgs)
        # logger.info (f"Role messages: {role_msgs} will be sent to {gpt_model}.")
        mistral_response = mist_client.chat(
            model=gpt_model,
            messages=role_msgs,
        )
            
        # print("Response received.")
        r_text = mistral_response.choices[0].message.content
        # print("The Returned text is as follows: ", r_text)
        return r_text
    
# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------
def main():
    print("This is a library of functions supporting turning transcripts into embeddings and storing them to a ChromaDB collection.")
    
if __name__ == "__main__":
    main()
    
