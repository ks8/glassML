# Parsing for CNN
from argparse import ArgumentParser, Namespace


def add_train_args(parser: ArgumentParser):
    """
    Add train args to an ArgumentParser.
    """

    # Create parser and add arguments
    parser.add_argument('--final_lr', type=float, default=1e-4, help='final learning rate')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to run')
    parser.add_argument('--beta', type=float, default=0.01, help='L2 regularization parameter')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--training_data_aug', action='store_true', help='Boolean for training dataset augmentation')
    parser.add_argument('--restart', action='store_true', help='Boolean for using a restart file')
    parser.add_argument('--restart_file', default="test.ckpt",
                        help='File name for previous saved model - just use <filename>.ckpt')
    parser.add_argument('--k_fold_split', action='store_true', default=False,
                        help='Split the data for nested k-fold cross validation')
    parser.add_argument('--num_folds', type=int, default=1, help='Number of folds when performing cross validation')
    parser.add_argument('--fold_index', type=int, default=1, help='Which fold to use for k-fold cross validation')
    parser.add_argument('--use_inner_test', action='store_true', default=False,
                        help='Use inner test set as test set during k-fold cross validation')
    parser.add_argument('--data_path', type=str, default=None, help='Path to metadata JSON')
    parser.add_argument('--im_size', type=int, default=250, help='Image size (one side length) in pixels')
    parser.add_argument('--seed', type=int, default=0, help='Random state seed')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory where model checkpoints will be saved')
    parser.add_argument('--val_test_size', type=float, default=0.2, help='Fraction of val + test data')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--max_iters', type=int, default=20,
                        help='Number of hyperparameter choices to try')
    parser.add_argument('--trials_dir', type=str, default=None,
                        help='(Optional) Path to a directory where previous trial results are written; hyperopt will'
                             'restart from this info')


def parse_train_args() -> Namespace:
    """
    Parses arguments for training.
    :return: Args
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    return args


