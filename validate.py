import os
import json

def get_filenames_in_dir(directory, suffix=None):
    filenames = []
    for file in os.listdir(directory):
        if file.endswith(suffix+".jpg" if suffix else ''):
            filenames.append(file)
    return set(filenames)

def validate_json_file(file_path, valid_keys):
    with open(file_path, 'r') as file:
        data = json.load(file)
    keys_in_json = set(data.keys())

    if keys_in_json == valid_keys:
        return True, None
    else:
        missing_keys = valid_keys - keys_in_json
        extra_keys = keys_in_json - valid_keys
        return False, (missing_keys, extra_keys)

def validate_file(file_path, verbose = True) -> bool:
    if "ipynb_checkpoint" in file_path:
        print(f"Not doing {file_path}")
        return True
    # Determine the directory to apply the correct validation rule
    directory = file_path.split(os.sep)[-2]
    round = file_path.split(os.sep)[-3]

    # Define directories for images
    image_dir = 'images/' if round!="second_round" else 'yesbut_second_round'
    image_split_dir = 'images_split/'

    # Determine the set of valid keys based on the directory
    if directory in ['whyfunny', 'punchline']:
        valid_keys = get_filenames_in_dir(image_dir)
    elif directory == 'left':
        valid_keys = get_filenames_in_dir(image_split_dir, '_YES')
    elif directory == 'right':
        valid_keys = get_filenames_in_dir(image_split_dir, '_BUT')
    else:
        print(f"Unknown directory type for file {file_path}")
        return False

    # Validate the file
    is_valid, diff = validate_json_file(file_path, valid_keys)
    if is_valid:
        if verbose:
            print(f"File {file_path} is valid.")
        return True
    else:
        missing_keys, extra_keys = diff
        print(f"File {file_path} is invalid.")
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if extra_keys:
            print(f"  Extra keys: {extra_keys}")
        return False

def validate_all_files():
    root_dir = 'outputs'
    directories = ['whyfunny', 'punchline', 'left', 'right']

    for directory in directories:
        full_dir_path = os.path.join(root_dir, directory)
        for subdir, _, files in os.walk(full_dir_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(subdir, file)
                    validate_file(file_path)

if __name__=="__main__":
    validate_all_files()