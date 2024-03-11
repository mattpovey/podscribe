import json
import chromadb
import pandas as pd
from libPodSemSearch import setup_chromadb, query_chromadb

# import config
from config import \
chromadb_name, \
chroma_collection, \
collection_metadata

# pod_prefix="rihPodcast"
# pod_dir=f'{pod_prefix}/audio'
# tscript_dir=f'{pod_prefix}/transcripts'
# csv_dir=f'{pod_prefix}/csv'
# Different lengths of max_tokens give different results in searches. The goal is to capture as much 
# meaning in each chunk without having too many different ideas in each chunk. 
# Multiple token length indexes can be created in ChromaDB. Setting max_tokens selects the appropriate
# index for searches run with this script. 

# max_tokens=240

# Setup ChromaDB Variables
# chromadb_name = f'{pod_prefix}/chroma.db'
# chroma_collection = f'{pod_prefix}_{max_tokens}T_Collection'
# collection_metadata = {"hnsw:space": "cosine", "model.max_seq_length": max_tokens}
print(chromadb_name)

num_results = 20

query = "British Prime Ministers"

chr_client, collection = setup_chromadb(chromadb_name, chroma_collection, collection_metadata)

results = query_chromadb(collection, query, num_results)
print(type(results))
for result in results:
    print(results[result][1])
    print(results[result][3])
    print(results[result][5])
#print(results)

# for i in results:
#     ep_id = results['ep_id']
#     print(f'Result episode number: {ep_id}')




# results = collection.query(
#     query_texts=[query],
#     n_results=25,
#     include = ["documents", "metadatas", "distances"]
# )

# # delete the 'embeddings' key from the results dict
# # The 'include' statement stops the embeddings values being returned but the key is still there with NONE as the value' 
# results.pop('embeddings', None)

# df_ord = pd.DataFrame()
# # Create columns for ids, distances, documents and metadata
# for key, value in results.items():
#     df_ord[key] = value[0]

# # Sort the results by distance
# df_ord = df_ord.sort_values(by='distances', ascending=False)
# df_ord.reset_index(drop=True, inplace=True)

# res_eps = {}
# for i in range(num_results):
#     ep_id = df_ord['ids'][i] #0
#     ep_title = df_ord['metadatas'][i]['title'] #1
#     ep_number = df_ord['metadatas'][i]['number'] #2
#     ep_date = df_ord['metadatas'][i]['date'] #3
#     ep_text = df_ord['documents'][i] #4
#     ep_url = df_ord['metadatas'][i]['url'] #5
#     ep_dist = df_ord['distances'][i] #6

#     res_eps[f'result_{i}'] = [ep_id, ep_number, ep_dist, ep_title, ep_date, ep_text, ep_url] 

# for i in res_eps:
#     print(f'Result episode number: {res_eps[i][1]}')
#     print(f'Similarity: {res_eps[i][2]}')
#     print(f'Title: {res_eps[i][3]}')
#     print(f'Released on: {res_eps[i][4]}')
#     print(f'Quote: {res_eps[i][5]}')
#     print(f'Listen here: {res_eps[i][6]}')
#     print('\n')
