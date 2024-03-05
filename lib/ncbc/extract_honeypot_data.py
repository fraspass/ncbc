"""Read raw ICL honeypot data logs and extract the commands from each session

Example usage:

"python3 extract_honeypot_data.py -e 1000"
- Extracts 1000 .zip files from raw honeypot directory and saves them to target directory

"python3 extract_honeypot_data.py -d -e"
- Deletes contents of target directory, then extracts all .zip files from honeypot directory

Note: If the number of .zip files is not specified, all files are extracted
"""

import os
import sys
import shutil
import json
import zipfile
from zipfile import ZipFile
# from tqdm import tqdm
from argparse import ArgumentParser
import traceback
import logging

raw_dir = '/home/hpms/icl'

# ------ CHANGE TO YOUR OWN DIRECTORY -----------
target_dir = '/home/dg3918/data/test'
# -----------------------------------------------

def empty_folder(dir):
    ls_dir = os.listdir(dir)
    print(f'Deleting contents of {dir}')
    # for f in tqdm(ls_dir):
    for f in ls_dir:
        shutil.rmtree(os.path.join(dir, f))
    print('Deletion complete.')

def extract_commands(N = None):
    """Read raw .zip session files, extract the Commands value from each session
    and write the Commands to files in personal directory
    """

    # List all .zip files in hpms directory
    ls_raw_dir = os.listdir(raw_dir)

    if N == None:
        N = len(ls_raw_dir)
    
    # Create data/sessions directory if it does not exist
    sessions_dir = os.path.join(target_dir, 'sessions')
    if not os.path.exists(sessions_dir):
        os.makedirs(sessions_dir)

    print('Extracting commands from raw session files...')
    d = 1 # session number
    # For each zip file, iterate through the files contained in the zip and extract the commands

    # for zip_file in tqdm(ls_raw_dir[::-1][:N]):
    for zip_file in ls_raw_dir[::-1][:N]:
        file_path = os.path.join(raw_dir, zip_file)
        if zipfile.is_zipfile(file_path):
            with ZipFile(file_path, 'r') as archive:
                for file in archive.namelist():
                    try:
                        session = json.load(archive.open(file))
                        commands = session['Commands']
                        if commands != []: # Only write to new file if Commands field is nonempty
                            with open(os.path.join(sessions_dir, str(d)), 'w') as f:
                                json.dump(commands, f)
                            d += 1

                    except Exception as e:
                        print(f'Error while reading file {file_path}')
                        logging.error(traceback.format_exc())
                        
    print(f'Extraction complete. {d} session files saved to {sessions_dir}')

def main(argv):
    parser = ArgumentParser()
    parser.add_argument('-d', '--delete', action='store_true',
                        help=f'Remove all saved data in {target_dir}')
    parser.add_argument('-e', '--extract', action='store_true',
                        help=f'Extract data in {raw_dir} and save to {target_dir}')
    parser.add_argument('N', nargs='?', type=int, default=None,
                        help='Number of .zip files to extract. If not specified, all files are processed')
    
    args = parser.parse_args()
    if args.delete:
        empty_folder(target_dir)
    if args.extract:
        extract_commands(args.N)

if __name__ == "__main__":
    main(sys.argv[1:])
