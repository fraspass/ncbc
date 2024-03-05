import os
from tqdm import tqdm
import json
import pandas as pd
from time import time
import re
import pickle
from clean_commands import *

if __name__ == '__main__':

    # Parameters for filtering extreme rare and common words
    no_below = 5
    no_above = 0.2

    # ------------ CHANGE TO YOUR OWN DIRECTORIES -----------------
    sessions_dir = '/Users/daniyar/Documents/m4r/data/sessions/' 
    save_dir = '/Users/daniyar/Documents/m4r/results/hpms/A/'
    # -------------------------------------------------------------

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ls_sessions_dir = os.listdir(sessions_dir)
    ls_sessions_dir.sort(key=int)

    D = len(ls_sessions_dir)
    print(D, 'session files')
    N_D = D

    # Load data
    print(f'Loading {N_D} session files from {sessions_dir} ...')
    data = {}
    for d in tqdm(ls_sessions_dir[:N_D]):
        session_file = os.path.join(sessions_dir, d)
        
        with open(session_file, 'rb') as f:
            session = json.load(f)
            session = [[tuple(session)]]
            data[d] = pd.DataFrame.from_records(session)

    # Merge all sessions
    print(f'Merging {N_D} sessions...')
    t1 = time()
    data = pd.concat(data)
    dt = time() - t1
    print(f'Merged sessions in {dt:0.2f}s')

    data.reset_index(drop=True, inplace=True)
    data = data.rename(columns={0:'Commands'})  

    # Keep only unique sessions
    data_unique = data.drop_duplicates().reset_index(drop=True)

    # Clean data: tokenize and split ';'-separated commands
    data_clean = clean_commands(data_unique['Commands'], no_below=1, no_above=1.1, url_hostnames=True)
    sessions_list = data_clean[0]
    dictionary = data_clean[1]
        
    # Remove empty commands ' ' and 'dot' commands '.'
    for i in tqdm(range(len(sessions_list))):
        sessions_list[i] = list(filter(lambda x: x != '', sessions_list[i]))
        sessions_list[i] = list(filter(lambda x: x != '.', sessions_list[i]))

    # Replace random HEX patterns 
    replacements = {r"(?<!\.)\bx[a-fA-F0-9]{2}\b(?!\.)": " HEX "}
    # Iterate through corpus
    for i in tqdm(range(len(sessions_list))):
        for j in range(len(sessions_list[i])):
            # Iterate through replacement patterns
            for key, value in replacements.items():
                text_test = re.sub(key, value, sessions_list[i][j])
                while text_test.startswith(" HEX "): 
                    text_test = text_test[1:] 
                while text_test.endswith(" HEX "):
                    text_test = text_test[:-1] 
            text_test = re.sub(' +', ' ', text_test) # detect double white spaces and substitute with single space
            sessions_list[i][j] = text_test

    # Replace common patterns
    # replace_dict = {r'(\w*%s\w*)'%'GHILIMEA': 'GHILIMEA_word'}
    replace_dict = {r'([a-zA-Z0-9_\.\-\*]*%s[a-zA-Z0-9_\.\-\*]*)'%'GHILIMEA': 'GHILIMEA_word'}
    for i in tqdm(range(len(sessions_list))):
        for j in range(len(sessions_list[i])):
            for key, value in replace_dict.items():
                sessions_list[i][j] = re.sub(key, value, sessions_list[i][j])

    # Obtain corpus and flattened list of commands
    commands_list = []
    corpus = []
    for session in sessions_list:
        corpus.append([])
        for command in session:
            c = command.split(' ')
            corpus[-1] += [c]
            commands_list += [c]

    # Remove single digit words and dots '.', '..' and '...'
    for i in tqdm(range(len(commands_list))):
        commands_list[i] = list(filter(lambda x: x != '.', commands_list[i]))
        commands_list[i] = list(filter(lambda x: x != '..', commands_list[i]))
        commands_list[i] = list(filter(lambda x: x != '...', commands_list[i]))
        commands_list[i] = list(filter(lambda x: len(x) > 1, commands_list[i]))

    # Dictionary from list of commands
    from gensim.corpora import Dictionary
    dictionary = Dictionary(commands_list)

    # Filter out extreme rare and common words
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    # Mapping between words and integers
    word_map = {}
    for w in range(len(dictionary)):
        word_map[w] = dictionary[w]
        word_map[dictionary[w]] = w

    # Transform words to integers
    W = {}
    i = 0
    for session in corpus:
        W[i] = {}
        j = 0
        for command in session:
            W[i][j] = []
            for word in command:
                try:
                    W[i][j] += [word_map[word]]
                except:
                    continue
            j += 1
        i += 1
    
    # Retain only unique sessions
    session_counter = {}
    rm_list = []
    sessions = []
    for s in W:
        session = []
        for c in W[s]:
            session.append(' '.join(str(x) for x in W[s][c]))
        session = ' '.join(str(x) for x in session[-1])
        session = session.strip(' ')
        sessions += [session]
        if session in session_counter:
            session_counter[session] += 1
            rm_list += [s]
        else:
            session_counter[session] = 1

    for s in rm_list:
        del W[s]
    
    # Remove empty commands
    for d in list(W.keys()):
        for j in list(W[d].keys()):
            if len(W[d][j])<1:
                del W[d][j]

    # Remove sessions containing less than two commands
    for d in list(W.keys()):
        if len(W[d]) < 2:
            del W[d]

    # Obtain filtered corpus
    W_filter = {}
    d = 0
    index_map = {}
    for s in W:
        j = 0
        W_filter[d] = {}
        for c in W[s]:
            W_filter[d][j] = W[s][c]
            index_map[d,j] = (s,c)
            j += 1
        d += 1
    
    del W

    # Save data

    # Filtered corpus
    W_name = 'W.pkl'
    W_path = os.path.join(save_dir, W_name)
    if os.path.exists(W_path):
        os.remove(W_path)
    with open(W_path, 'wb') as f:
        pickle.dump(W_filter, f)

    print(f'Filtered corpus saved to {W_path}')

    # Word and index maps
    maps = {}
    maps['word'] = word_map
    maps['index'] = index_map
    map_name = 'maps.pkl'
    map_path = os.path.join(save_dir, map_name)
    if os.path.exists(map_path):
        os.remove(map_path)
    with open(map_path, 'wb') as f:
        pickle.dump(maps, f)

    print(f'Word and index maps saved to {map_path}')
