# Create data folder with metadata
import json
import os
import argparse


def process_metadata(args):
    """
    Create metadata folder and file.
    :param args: Folder name info
    """

    data = []
    for root, subfolders, files in os.walk(args.data_dir):
        for f in files:
            # Set path
            path = os.path.join(root, f)

            # Set class label
            if args.category_label_1 in f:
                material = args.category_label_1
            elif args.category_label_2 in f:
                material = args.category_label_2

            # Set uid
            if args.software == 'DASH':
                uid_start_index = f.find('--') + 2
                uid_end_index = f.find('_normalized')
                uid = f[uid_start_index:uid_end_index]
            elif args.software == 'LAMMPS':
                uid_start_index = f.find('.') + 1
                uid = f[uid_start_index:uid_start_index + 5]
            data.append({'path': path, 'label': material, 'uid': uid})
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    json.dump(data, open(args.out_dir+'/'+args.out_dir+'.json', 'w'), indent=4, sort_keys=True)


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing data')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Directory name for metadata')
    parser.add_argument('--category_label_1', type=str, dest='category_label_1', default=None,
                        help='Name of one of two class labels')
    parser.add_argument('--category_label_2', type=str, dest='category_label_2', default=None,
                        help='Name of the other class label')
    parser.add_argument('--software', type=str, dest='software', default=None,
                        help='DASH or LAMMPS')

    args = parser.parse_args()
    process_metadata(args)


if __name__ == "__main__":
    main()
