#! /usr/bin/env python3
from nltk.tokenize import RegexpTokenizer
from gensim.corpora import Dictionary
import urllib

def clean_commands(dat, no_below=1, no_above=1.1, url_hostnames=True):
    """
    This function 
    1. splits multiple commands in the same line
    2. tokenize the commands
    3. replace rare commands by rarecommand

    :param dat: dataset
    :param no_below: Keep tokens which are contained in at least no_below documents.
    :param no_above: Keep tokens which are contained in no more than no_above documents (fraction of total corpus size, not an absolute number).

    :return sessins_token_list: tokenized list of sessions of commands
    :return dictionary: dictionary generated
    """
    # for commands splitted by ;
    sessions = []
    for session in dat:
        sessions.append([]) # to make list of lists
        for command in session:
            sessions[-1] += command.split('; ')
    # tokenizer
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9_\.\-\*]+')
    sessions_list = []
    commands_list = []
    for session in sessions:
        sessions_list.append([])
        commands_list.append([])
        for command in session:
            if url_hostnames and 'http' in command:
                    command_split = command.split(' ')
                    command_new = ''
                    for word in command_split:
                        if 'http' in word:
                            command_new += str(urllib.parse.urlparse(word).hostname)
                        else:
                            command_new += word
                        command_new += ' '
                    command_token = tokenizer.tokenize(command_new)
            else:
                command_token = tokenizer.tokenize(command)
            sessions_list[-1] += [command_token]
            commands_list[-1] += command_token
    dictionary = Dictionary(commands_list) 
    dictionary.filter_extremes(no_below, no_above)
    # Replace rare commands by rarecommand
    dictionary.id2token[-1] = 'rarecommand' # Add a 'rarecommand' token to the corpus and assign a unique value to it -1
    ''' -1 value indicates rareword, so when a token appears in sessions
    that is not in final dictionary (after filtering) it substitutes the unique value -1 to this token '''
    dictionary.token2id['rarecommand'] = -1 
    sessions_token_list = []
    for session in sessions_list:
        sessions_token_list.append([])
        commands_token_list = []
        for command in session:
            idxs = dictionary.doc2idx(command) # Gives to tokens of command the unique values/ids assigned
            commands_token_list.append(' '.join([dictionary[idx] for idx in idxs]))
        sessions_token_list[-1] += commands_token_list
    return sessions_token_list, dictionary
