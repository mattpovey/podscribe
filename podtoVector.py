from libPodSemSearch import *
# import chromadb

# Import directories from config.py
from config import pod_prefix, \
    pod_dir, \
    tscript_dir, \
    save_dir, \
    episode_dir, \
    wav_dir, \
    csv_dir, \
    transcription_dir

# Import filenames from config
from config import \
    metadata_csv, \
    raw_csv, \
    chunks_csv, \
    embeds_csv, \
    final_csv

# Import model params
from config import \
    max_tokens, \
    embeds_model, \
    embeds_model_cost, \
    test_model, \
    chromadb_name, \
    chroma_collection, \
    collection_metadata

# pod_prefix="rihPodcast"
# pod_dir=f'{pod_prefix}/audio'
# tscript_dir=f'{pod_prefix}/transcripts'
# csv_dir=f'{pod_prefix}/csv'
# Different lengths of max_tokens give different results in searches. The goal is to capture as much 
# meaning in each chunk without having too many different ideas in each chunk. 
# The current value of 350 is probably too high (likely twice as big as ideal) but for the sake of 
# speed, larger values are better. 
# max_tokens=240

# embeds_model="text-embeddings-ada-002"
# embeds_model="all-MiniLM-L6-v2"
# test_model="gpt3.5-turbo" # used to check validity of API key
# embeds_model_cost = 0 # per 1K tokens

# Setup ChromaDB Variables
# chromadb_name = f'{pod_prefix}/chroma.db'
# chroma_collection = f'{pod_prefix}_{max_tokens}T_Collection'
# collection_metadata = {"hnsw:space": "cosine", "model.max_seq_length": "240"}

# Setup filenames
# metadata_csv  =f'{csv_dir}/episodeMetadata.csv'
# raw_csv = f'{csv_dir}/raw.csv'
# chunks_csv = f'{csv_dir}/chunks_{str(max_tokens)}T.csv'
# embeds_csv = f'{csv_dir}/embeds{str(max_tokens)}T.csv'
# final_csv = f'{csv_dir}/embeds_url_{str(max_tokens)}T.csv'

# -----------------------------------------------------------------------------
# 1. Podcast processing
# * Check for new transcripts
# * Process into srt lines and save to a dataframe and csv
# -----------------------------------------------------------------------------
print('Checking for new episode transcripts')
df_raw = process_srt_files(tscript_dir, raw_csv, embeds_model)
check_raw(df_raw, tscript_dir, embeds_model, embeds_model_cost)

# Switch from raw_csv to chunks_csv
output_csv = chunks_csv

# Combine the text from the raw csv into chunks of max_tokens
print(f'\n Combining text into chunks of {max_tokens} tokens\n')
df_chunks = pd.DataFrame()
df_chunks, new_files = combine_text(raw_csv, chunks_csv, max_tokens, embeds_model)
if len(new_files) == 0:
    pass
else:
    print(f'New files to process: {len(new_files)}')
    # Check that only the embeddings column contains nan
    nan_check(df_chunks)
    save_csv(df_chunks, output_csv)

    # Add metadata to the chunks csv
    # THIS IS BROKEN FOR ADDITION OF NEW FILES
    print(f'\nAdding metadata to {output_csv}\n')
    try:
        df_chunks = add_metadata(output_csv, metadata_csv, embeds_model, feed_url)
    except ValueError as e:
        print(f'Error adding metadata: {e}')
        exit()

    save_csv(df_chunks, output_csv)

# -----------------------------------------------------------------------------
# 2. Prepare the data for storage in chromaDB
# -----------------------------------------------------------------------------
# Check that we only have new files that are not already in the chromaDB collection
df_chunks = incremendal_add(df_chunks, chromadb_name, chroma_collection, collection_metadata, chunks_csv)

if df_chunks is None:
    print('No files to add.')
    exit()
else:
    # Create a dictionary of the metadata for use in the ChromaDB metadatas field
    # Create a list of the text fields in a variable called 'documents'
    print(f"\nCreating documents list\n")
    chromadb_documents = df_chunks['text'].tolist()
    print(chromadb_documents[0])

    # Create a list of dicts with the title, number, date, filename, srt_index, tc_start, tc_end and url fields 
    # in a variable called 'metadata'. Each dict should contain all values for a row in name:value pairs.
    print(f"\nCreating metadata list\n")
    chromadb_metadata = []
    for index, row in df_chunks.iterrows():
        print(f'Adding metadata for {index}')
        clear_stdout()
        chromadb_metadata.append({'title': row['title'], 'number': row['number'], 'date': row['date'], 'filename': row['filename'], 'tokens': row['tokens'], 'srt_index': row['srt_index'], 'tc_start': row['tc_start'], 'tc_end': row['tc_end'], 'url': row['url']})
    #print(chromadb_metadata[:2])

    # Create a list of row IDsin a variable called 'rih_ids'
    print(f"\nCreating rih_ids list\n")
    chromadb_ids = []
    n=1
    for index, row in df_chunks.iterrows():
        chromadb_ids.append(str(n))
        n+=1
    print(chromadb_ids[:2])

# -----------------------------------------------------------------------------
# 3. Setup chromaDB
# -----------------------------------------------------------------------------

chr_client, collection = setup_chromadb(chromadb_name, chroma_collection, collection_metadata)
#chr_client.delete_collection(name=chroma_collection)
#exit()

# -----------------------------------------------------------------------------
# 4. Add the data to chromaDB
# -----------------------------------------------------------------------------
# Add the documents, embeddings, metadata and ids to the collection
print(f"\nAdding {len(chromadb_documents)} documents to {chroma_collection}\n")
tot_docs = len(chromadb_documents)
# Add the documents, embeddings, metadata and ids to the collection in groups of 100
n = 0
doc_count = 0
while n <= tot_docs:
    if n+100 >= tot_docs:
        collection.add(
            documents=chromadb_documents[n:],
            metadatas=chromadb_metadata[n:],
            ids=chromadb_ids[n:]
        )
    else:
        collection.add(
            documents=chromadb_documents[n:n+100],
            metadatas=chromadb_metadata[n:n+100],
            ids=chromadb_ids[n:n+100]
        )
    n+=100
    doc_count+=100
    clear_stdout()
    print(f'Added {doc_count} of {len(chromadb_documents)} documents to {chroma_collection}')
    print(f'There are {collection.count()} documents in {chroma_collection}')






