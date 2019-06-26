from argparse import ArgumentParser, Namespace
import os
from tempfile import TemporaryDirectory

import torch


def add_predict_args(parser: ArgumentParser):
    """Add predict arguments to an ArgumentParser."""
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--test_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--write_smiles', action='store_true', default=False,
                        help='Whether to write smiles in addition to writing predicted values')
    parser.add_argument('--preds_path', type=str,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')


def add_train_args(parser: ArgumentParser):
    """Add training arguments to an ArgumentParser."""
    # General arguments
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file, or to directory of files if pre-chunked')
    parser.add_argument('--train_data_path', type=str, help='Path to .json file containing train metadata')
    parser.add_argument('--val_data_path', type=str, help='Path to .json file containing validation metadata')
    parser.add_argument('--test_data_path', type=str, help='Path to .json file containing test metadata')
    parser.add_argument('--num_neighbors', type=int, help='Number of nearest neighbors used to build graphs')
    parser.add_argument('--augmentation_length', type=float, default=0.1, help='Window length for augmentation')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--vocab_path', type=str,
                        help='Path to .vocab file if using jtnn')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=['morgan', 'morgan_count', 'rdkit_2d', 'rdkit_2d_normalized', 'mordred'],
                        help='Method of generating additional features')  # TODO allow multiple options
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--predict_features', action='store_true', default=False,
                        help='Pre-train by predicting the additional features rather than the task values')
    parser.add_argument('--predict_features_and_task', action='store_true', default=False,
                        help='Pre-train by predicting the additional features in addition to the task values')
    parser.add_argument('--task_weight', type=float, default=1.0,
                        help='Weighting for the real tasks when also predicting features in multitask setting')                    
    parser.add_argument('--additional_atom_features', type=str, nargs='*', choices=['functional_group'], default=[],
                        help='Use additional features in atom featurization')
    parser.add_argument('--functional_group_smarts', type=str, default='chemprop/features/smarts.txt',
                        help='Path to txt file of smarts for functional groups, if functional_group features are on.')
    parser.add_argument('--sparse', action='store_true', default=False,
                        help='Store features as sparse (can save memory for sparse features')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--load_encoder_only', action='store_true', default=False,
                        help='If a checkpoint_dir is specified for training, only loads weights from encoder'
                             'and not from the final feed-forward network')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression', 'regression_with_binning',
                                 'unsupervised', 'bert_pretraining', 'kernel'],
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.'
                             'Unsupervised means using Caron et al pretraining (from FAIR).'
                             'bert_pretraining means using BERT (Devlin et al) style pretraining (from Google)'
                             'kernel means using some kind of kernel pretraining')
    parser.add_argument('--unsupervised_n_clusters', type=int, default=10000,
                        help='Number of clusters to use for unsupervised learning labels')
    parser.add_argument('--prespecified_chunks_max_examples_per_epoch', type=int, default=1000000,
                        help='When using prespecified chunks, load up to this many examples per "epoch"')
    parser.add_argument('--num_bins', type=int, default=20,
                        help='Number of bins for regression with binning')
    parser.add_argument('--num_chunks', type=int, default=1,
                        help='Specify > 1 if your dataset is really big')
    parser.add_argument('--chunk_temp_dir', type=str, default='temp_chunks',
                        help='temp dir to store chunks in')
    parser.add_argument('--memoize_chunks', action='store_true', default=False,
                        help='store memo dicts for mol2graph in chunk_temp_dir when chunking, at large disk space cost')
    parser.add_argument('--separate_val_set', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_set_features', type=str, nargs='*',
                        help='Path to .pckl file with features for separate val set')
    parser.add_argument('--separate_test_set', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_set_features', type=str, nargs='*',
                        help='Path to .pckl file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold', 'scaffold_balanced', 'scaffold_one', 'scaffold_overlap', 'predetermined'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_test_by_overlap_dataset', type=str,
                        help='Dataset to use to split test set by overlap')
    parser.add_argument('--scaffold_overlap', type=float, default=None,
                        help='Proportion of molecules in val/test sets which should contain scaffolds in the train set'
                             'For use when split_type == "scaffold_overlap"')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--k_fold_split', action='store_true', default=False,
                        help='Split the data for nested k-fold cross validation')
    parser.add_argument('--fold_index', type=int, default=None,
                        help='Which fold to use for k-fold cross validation')
    parser.add_argument('--val_test_size', type=float, default=0.2, help='Fraction of val + test data')
    parser.add_argument('--use_inner_test', action='store_true', default=False,
                        help='Use inner test set as test set during k-fold cross validation')
    parser.add_argument('--num_val_runs', type=int, default=1,
                        help='Number of times to iterate through validation set during training '
                             '(i.e. with data augmentation)')
    parser.add_argument('--num_test_runs', type=int, default=1,
                        help='Number of times to iterate through test set during testing '
                             '(i.e. with data augmentation)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'r2', 'accuracy', 'argmax_accuracy', 'log_loss',
                                 'majority_baseline_accuracy'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--labels_to_show', type=str, nargs='+',
                        help='List of targets to show individual scores for, if specified')
    parser.add_argument('--max_data_size', type=int, default=None,
                        help='Maximum number of data points to load')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to run processes sequentially instead of in parallel'
                             '(currently only effects vocabulary for bert_pretraining,'
                             'and soon will affect featurization in main training loop)')
    parser.add_argument('--parallel_featurization', action='store_true', default=False,
                        help='asychronous featurization for significant speedup; will become default in future')
    parser.add_argument('--batch_queue_max_size', type=int, default=1,
                        help='Maximum size of queue of batches for asynchronous featurization')
    parser.add_argument('--batches_per_queue_group', type=int, default=200,
                        help='Number of batches to get at a time from the asynchronous featurizer')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--skip_smiles_path', type=str,
                        help='Path to a data .csv containing smiles that should NOT be trained on')
    parser.add_argument('--keep_nan_metrics', action='store_true', default=False,
                        help='AUC metrics can return nan when the test set for a target is all 0 or all 1.'
                             'Turning this on defaults the value to 0.5 rather than skipping that target')
    parser.add_argument('--test_split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split (train/val/test) to use as the test set in run_training.py')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--truncate_outliers', action='store_true', default=False,
                        help='Truncates outliers in the training set to improve training stability'
                             '(All values outside mean +/- 3 * std are truncated to equal mean +/- 3 * std)')
    parser.add_argument('--warmup_epochs', type=float, nargs='*', default=[2.0],
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help='learning rate optimizer')
    parser.add_argument('--scheduler', type=str, default='noam', choices=['noam', 'none', 'decay'],
                        help='learning rate scheduler')
    parser.add_argument('--separate_ffn_lr', action='store_true', default=False,
                        help='Whether to use a separate optimizer/lr scheduler for the ffn'
                             'rather than sharing optimizer/scheduler with the message passing encoder')
    parser.add_argument('--uniform_init', action='store_true', default=False,
                        help='Initialize with xavier_uniform instead of xavier_normal, with proper scaling for relu')
    parser.add_argument('--init_lr', type=float, nargs='*', default=[1e-4],
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, nargs='*', default=[1e-3],
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, nargs='*', default=[1e-4],
                        help='Final learning rate')
    parser.add_argument('--lr_scaler', type=float, nargs='*', default=[1.0],
                        help='Amount by which to scale init_lr, max_lr, and final_lr (for convenience)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='lr decay per epoch, for decay scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Maximum gradient norm when performing gradient clipping')
    parser.add_argument('--weight_decay', type=float, nargs='*', default=[0.0],
                        help='L2 penalty on optimizer to keep parameter norms small')
    parser.add_argument('--no_target_scaling', action='store_true', default=False,
                        help='Turn off scaling of regression targets')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--adjust_weight_decay', action='store_true', default=False,
                        help='Adjust weight decay dynamically, with init = args.weight_decay'
                             'to try to preserve pnorm around initialization level')
    parser.add_argument('--adjust_weight_decay_step', type=float, default=1e-5,
                        help='How much to add/subtract from weight decay at a time, when adjusting')
    parser.add_argument('--bert_mask_prob', type=float, default=0.15,
                        help='Probability of masking when dataset_type == "bert_pretraining"')
    parser.add_argument('--bert_vocab_func', type=str, default='feature_vector',
                        choices=['atom', 'atom_features', 'feature_vector', 'substructure'],
                        help='Vocab function when dataset_type == "bert_pretraining"')
    parser.add_argument('--bert_max_vocab_size', type=int, default=0,
                        help='Set to > 0 to limit vocab size, replacing others with unk token')
    parser.add_argument('--bert_smiles_to_sample', type=int, default=10000,
                        help='Set to > 0 to limit sampled smiles for vocab computation')
    parser.add_argument('--bert_substructure_sizes', type=int, nargs='+', default=[3],
                        help='Size of substructures to mask when bert_vocab_func == "substructure"')
    parser.add_argument('--bert_mask_type', type=str, default='cluster',
                        choices=['random', 'correlation', 'cluster'],
                        help='How to mask atoms in bert_pretraining')
    parser.add_argument('--bert_mask_bonds', action='store_true', default=False,
                        help='mask bonds in bert pretraining when both adjacent atoms are zeroed out')
    parser.add_argument('--additional_output_features', type=str, nargs='*', choices=['functional_group'], default=[],
                        help='Use additional features in bert output features to predict,'
                             'but not in original input atom features. Only supported for bert_vocab_func = feature_vector.')
    parser.add_argument('--kernel_func', type=str,
                        choices=['features', 'features_dot', 'WL'],
                        help='Kernel function for kernel pretraining')
    parser.add_argument('--last_batch', action='store_true', default=False,
                        help='Whether to include the last batch in each training epoch even if'
                             'it\'s less than the batch size')
    parser.add_argument('--class_balance', action='store_true', default=False,
                        help='Whether to enforce class balance by reweighting the loss based on class size'
                             '(for classification datasets only)')

    # Model arguments
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--diff_depth_weights', action='store_true', default=False,
                        help='Whether to use a different weight matrix at each step of message passing')
    parser.add_argument('--layers_per_message', type=int, default=1,
                        help='Num linear layers between message passing steps')
    parser.add_argument('--layer_norm', action='store_true', default=False,
                        help='Add layer norm after each message passing step')
    parser.add_argument('--normalize_messages', action='store_true', default=False,
                        help='Normalize bond messages at each message passing step')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh'],
                        help='Activation function')
    parser.add_argument('--attention', action='store_true', default=False,
                        help='Perform self attention over the atoms in a molecule')
    parser.add_argument('--message_attention', action='store_true', default=False,
                        help='Perform attention over messages.')
    parser.add_argument('--global_attention', action='store_true', default=False,
                        help='True to perform global attention across all messages on each message passing step')
    parser.add_argument('--message_attention_heads', type=int, default=1,
                        help='Number of heads to use for message attention')
    parser.add_argument('--attention_pooling', action='store_true', default=False,
                        help='Perform multi-headed attention pooling over the atoms in a molecule at the end of '
                             'message passing')
    parser.add_argument('--bond_attention', action='store_true', default=False,
                        help='Perform multi-headed attention over the bonds in a molecule during message passing')
    parser.add_argument('--bond_attention_pooling', action='store_true', default=False,
                        help='Perform multi-headed attention pooling over the bonds in a molecule at the end of '
                             'message passing')
    parser.add_argument('--attention_pooling_heads', type=int, default=1,
                        help='Number of heads to use for multi-headed attention pooling')
    parser.add_argument('--attention_viz', action='store_true', default=False,
                        help='Visualizes multi-headed attention pooling')
    parser.add_argument('--master_node', action='store_true', default=False,
                        help='Add a master node to exchange information more easily')
    parser.add_argument('--master_dim', type=int, default=600,
                        help='Number of dimensions for master node state')
    parser.add_argument('--use_master_as_output', action='store_true', default=False,
                        help='Use master node state as output')
    parser.add_argument('--addHs', action='store_true', default=False,
                        help='Explicitly adds hydrogens to the molecular graph')
    parser.add_argument('--three_d', action='store_true', default=False,
                        help='Adds 3D coordinates to atom and bond features')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')                     
    parser.add_argument('--virtual_edges', action='store_true', default=False,
                        help='Adds virtual edges between non-bonded atoms')
    parser.add_argument('--drop_virtual_edges', action='store_true', default=False,
                        help='Randomly drops O(n_atoms) virtual edges so O(n_atoms) edges total instead of O(n_atoms^2)')
    parser.add_argument('--learn_virtual_edges', action='store_true', default=False,
                        help='Learn which virtual edges to add, limited to O(n_atoms)')
    parser.add_argument('--deepset', action='store_true', default=False,
                        help='Modify readout function to perform a Deep Sets set operation using linear layers')
    parser.add_argument('--set2set', action='store_true', default=False,
                        help='Modify readout function to perform a set2set operation using an RNN')
    parser.add_argument('--set2set_iters', type=int, default=3,
                        help='Number of set2set RNN iterations to perform')
    parser.add_argument('--jtnn', action='store_true', default=False,
                        help='Build junction tree and perform message passing over both original graph and tree')
    parser.add_argument('--ffn_input_dropout', type=float, default=None,
                        help='Input dropout for higher-capacity FFN (defaults to dropout)')
    parser.add_argument('--ffn_dropout', type=float, default=None,
                        help='Dropout for higher-capacity FFN (defaults to dropout)')
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--adversarial', action='store_true', default=False,
                        help='Adversarial scaffold regularization')
    parser.add_argument('--wgan_beta', type=float, default=10,
                        help='Multiplier for WGAN gradient penalty')
    parser.add_argument('--gan_d_per_g', type=int, default=5,
                        help='GAN discriminator training iterations per generator training iteration')
    parser.add_argument('--gan_lr_mult', type=float, default=0.1,
                        help='Multiplier for GAN generator learning rate')
    parser.add_argument('--gan_use_scheduler', action='store_true', default=False,
                        help='Use noam scheduler for GAN optimizers')
    parser.add_argument('--maml', action='store_true', default=False,
                        help='Use model agnostic meta learning')
    parser.add_argument('--maml_lr', type=float, default=0.01,
                        help='MAML SGD lr')
    parser.add_argument('--maml_batches_per_epoch', type=int, default=5,
                        help='Number of batches of maml to perform during each training epoch')
    parser.add_argument('--maml_batch_size', type=int, default=5,
                        help='maml batch size, i.e. number of tasks to sample, train, and compute test loss on'
                             'before summing test losses to update meta weights')
    parser.add_argument('--moe', action='store_true', default=False,
                        help='Use mixture of experts model')
    parser.add_argument('--cluster_split_seed', type=int, default=0,
                        help='Random seed for K means cluster split')
    parser.add_argument('--cluster_max_ratio', type=float, default=4,
                        help='Max ratio of sizes between two clusters for K means cluster split')
    parser.add_argument('--batch_domain_encs', action='store_true', default=False,
                        help='compute domain encoding means in batches, for speed')
    parser.add_argument('--lambda_moe', type=float, default=0.1,
                        help='Multiplier for moe vs mtl loss')
    parser.add_argument('--lambda_critic', type=float, default=1.0,
                        help='Multiplier for critic loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.001,
                        help='Multiplier for entropy regularization')
    parser.add_argument('--m_rank', type=int, default=100,
                        help='Mahalanobis matrix rank in moe model')
    parser.add_argument('--num_sources', type=int, default=10,
                        help='Number of source tasks for moe')
    parser.add_argument('--mayr_layers', action='store_true', default=False,
                        help='Use Mayr et al versions of dropout and linear layers (diff is bias unit scaling)')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Whether to freeze the layers of the message passing encoder')
    parser.add_argument('--gradual_unfreezing', action='store_true', default=False,
                        help='Unfreeze layers one at a time starting from the end, when using pretrained init')
    parser.add_argument('--epochs_per_unfreeze', type=int, default=1,
                        help='Number of epochs between unfreezing layers when doing gradual unfreezing')
    parser.add_argument('--discriminative_finetune', action='store_true', default=False,
                        help='Smaller LR for earlier layers during finetuning')
    parser.add_argument('--discriminative_finetune_decay', type=float, default=0.4,
                        help='Decay per param group when doing discriminative finetuning')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--no_bond_features', action='store_true', default=False,
                        help='Don\'t use bond features (only atom features)')


def update_checkpoint_args(args: Namespace):
    """Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size."""
    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname == 'model.pt':
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)

    if args.ensemble_size == 0:
        raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')


def modify_predict_args(args: Namespace):
    assert args.test_path
    assert args.preds_path
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None

    update_checkpoint_args(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # Create directory for preds path
    preds_dir = os.path.dirname(args.preds_path)
    if preds_dir != '':
        os.makedirs(preds_dir, exist_ok=True)


def parse_predict_args() -> Namespace:
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    modify_predict_args(args)

    return args


def modify_train_args(args: Namespace):
    """Modifies and validates training arguments."""
    global temp_dir  # Prevents the temporary directory from being deleted upon function return

    assert args.data_path is not None
    assert args.dataset_type is not None

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        temp_dir = TemporaryDirectory()
        args.save_dir = temp_dir.name

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.target_scaling = not args.no_target_scaling
    del args.no_target_scaling

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'unsupervised':
            args.metric = 'log_loss'
        elif args.dataset_type == 'bert_pretraining':
            if args.bert_vocab_func == 'feature_vector':
                args.metric = 'rmse'
            else:
                if args.metric not in ['log_loss', 'argmax_accuracy', 'majority_baseline_accuracy']:
                    args.metric = 'log_loss'
        elif args.dataset_type == 'kernel':
            if args.kernel_func in ['features', 'features_dot', 'WL']:  # could have other kernel_funcs with different metrics
                args.metric = 'rmse'
            else:
                raise ValueError(f'metric not implemented for kernel function "{args.kernel_func}".')
        else:
            args.metric = 'rmse'

    if not (args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc', 'accuracy'] or
            (args.dataset_type == 'regression' or args.dataset_type == 'regression_with_binning') and args.metric in ['rmse', 'mae', 'r2'] \
            or args.dataset_type == 'kernel' and args.metric in ['rmse']) \
            and not args.dataset_type in ['unsupervised', 'bert_pretraining']:
        raise ValueError(f'Metric "{args.metric}" invalid for dataset type "{args.dataset_type}".')

    args.minimize_score = args.metric in ['rmse', 'mae', 'log_loss']

    update_checkpoint_args(args)

    if args.jtnn:
        if not hasattr(args, 'vocab_path'):
            raise ValueError('Must provide vocab_path when using jtnn')
        elif not os.path.exists(args.vocab_path):
            raise ValueError(f'Vocab path "{args.vocab_path}" does not exist')

    args.use_input_features = args.features_generator or args.features_path

    if args.predict_features_and_task:
        assert args.dataset_type == 'regression'
        args.predict_features = True
        
    if args.predict_features or args.kernel_func in ['features', 'features_dot']:
        assert args.features_generator or args.features_path
        args.use_input_features = False

    if args.features_generator is not None and 'rdkit_2d_normalized' in args.features_generator:
        assert not args.features_scaling

    if args.dataset_type == 'bert_pretraining':
        assert not args.features_only
        args.use_input_features = False

    if args.dataset_type == 'unsupervised':
        args.separate_ffn_lr = True

    args.num_lrs = 1 + args.separate_ffn_lr
    lr_params = [args.init_lr, args.max_lr, args.final_lr, args.lr_scaler, args.warmup_epochs, args.weight_decay]
    for lr_param in lr_params:
        assert 1 <= len(lr_param) <= args.num_lrs
        if args.separate_ffn_lr:
            if len(lr_param) == 1:
                lr_param *= 2

    for i in range(args.num_lrs):
        args.init_lr[i] *= args.lr_scaler[i]
        args.max_lr[i] *= args.lr_scaler[i]
        args.final_lr[i] *= args.lr_scaler[i]

    del args.lr_scaler

    if args.dataset_type != 'kernel':
        assert args.ffn_num_layers >= 1

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size
    if args.ffn_input_dropout is None:
        args.ffn_input_dropout = args.dropout
    if args.ffn_dropout is None:
        args.ffn_dropout = args.dropout

    assert (args.split_type == 'scaffold_overlap') == (args.scaffold_overlap is not None)
    assert (args.split_type == 'predetermined') == (args.folds_file is not None) == (args.test_fold_index is not None)

    if args.test:
        args.epochs = 0

    # if os.path.isdir(args.data_path):  # gave a directory of chunks instead of just one file; will load a few per epoch
    #     args.prespecified_chunk_dir = args.data_path
    #     for root, _, names in os.walk(args.data_path):
    #         args.data_path = os.path.join(root, names[0])  # just pick any one for now, for preprocessing
    #         break
    # else:
    #     args.prespecified_chunk_dir = None

    if args.class_balance:
        assert args.dataset_type == 'classification'
    
    # TODO uncomment when this becomes the default, and remove the parallel_featurization option
    # args.parallel_featurization = (not args.maml and not args.moe and not args.sequential)


def parse_train_args() -> Namespace:
    """Parses arguments for training (includes modifying/validating arguments)."""
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args
