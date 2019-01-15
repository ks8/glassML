# Create data folder with metadata
import json
import os
import re
import argparse


def process_metadata(args):

    data = []
    for root, subfolders, files in os.walk(args.data_dir):
        for f in files:
            path = os.path.join(root, f)
            category = re.findall(r'glass', path)
            if len(category) == 1:
                material = 'glass'
            else:
                material = 'liquid'
            data.append({'path': path, 'label': material})
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    json.dump(data, open(args.out_dir+'/'+args.out_dir+'.json', 'w'), indent=4, sort_keys=True)


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir', type=str, dest='data_dir', default=None, help='Directory containing data')
    parser.add_argument('-out_dir', type=str, dest='out_dir', default=None, help='Directory name for metadata')

    args = parser.parse_args()
    process_metadata(args)


if __name__ == "__main__":
    main()
