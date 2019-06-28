# Main function for training a CNN
import os
from parsing import parse_train_args
import tensorflow as tf
import numpy as np
import json
from generator import Generator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from tqdm import tqdm
from pprint import pformat
from create_logger import create_logger
from data_utils import get_unique_labels, create_one_hot_mapping, convert_to_one_hot
from model import model


def run_training(args, logger):
    """
    Run training.
    :param args: arg info
    :return:
    """

    # Set up logger
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Print args
    debug(pformat(vars(args)))

    # Load metadata and convert to one-hot labels
    debug('Loading data')
    metadata = json.load(open(args.data_path, 'r'))
    unique_labels = get_unique_labels(metadata)
    one_hot_mapping = create_one_hot_mapping()
    metadata = convert_to_one_hot(metadata, one_hot_mapping)

    # Define input and output sizes
    im_size = args.im_size
    n_outputs = len(unique_labels)

    # Train/val/test split
    if args.k_fold_split:
        data_splits = []
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        for train_index, test_index in kf.split(metadata):
            splits = [train_index, test_index]
            data_splits.append(splits)
        data_splits = data_splits[args.fold_index]

        if args.use_inner_test:
            train_indices, remaining_indices = train_test_split(data_splits[0], test_size=args.val_test_size,
                                                                random_state=args.seed)
            validation_indices, test_indices = train_test_split(remaining_indices, test_size=0.5,
                                                                random_state=args.seed)
    
        else:
            train_indices = data_splits[0]
            validation_indices, test_indices = train_test_split(data_splits[1], test_size=0.5, random_state=args.seed)

        train_metadata = list(np.asarray(metadata)[list(train_indices)])
        validation_metadata = list(np.asarray(metadata)[list(validation_indices)])
        test_metadata = list(np.asarray(metadata)[list(test_indices)])

    else:
        train_metadata, remaining_metadata = train_test_split(metadata, test_size=args.val_test_size,
                                                              random_state=args.seed)
        validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=args.seed)

    # Dataset lengths
    train_data_length, val_data_length, test_data_length = \
        len(train_metadata), len(validation_metadata), len(test_metadata)
    debug('train size = {:,} | val size = {:,} | test size = {:,}'.format(
        train_data_length,
        val_data_length,
        test_data_length)
    )

    # Batch generators
    train_generator = Generator(train_metadata, im_size=im_size, num_channel=4)
    validation_generator = Generator(validation_metadata, im_size=im_size, num_channel=4)
    test_generator = Generator(test_metadata, im_size=im_size, num_channel=4)

    # Define batches per epoch and examples per eval
    batches_per_epoch = int(float(len(train_metadata))/float(args.batch_size))

    # Build the graph for the deep net
    x = tf.placeholder(tf.float32, [None, im_size, im_size, 3])
    y_ = tf.placeholder(tf.float32, [None, n_outputs])
    y_conv, dropout, w_conv1, w_conv2, w_fc1, w_fc2 = model(x, n_outputs)

    # Define the loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    regularizers = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(w_fc2)
    loss = tf.reduce_mean(loss + args.beta*regularizers)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Save GPU memory preferences
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Saver
    saver = tf.train.Saver()

    # AUC tracker
    auc_tracker = 0.0
    best_epoch = 0

    # Run the network
    with tf.Session(config=config) as sess:

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Restore a previous model
        if args.restart:
            saver.restore(sess, args.restart_file)

        # Training
        debug('Training')
        for epoch in range(args.epochs):
            debug('epoch {}'.format(epoch))
            debug('Evaluating')

            # Evaluate AUC on validation set
            validation_preds = np.array([[0, 0]])
            validation_labels = np.array([[0, 0]])
            for validation_x, validation_y in validation_generator.data_in_batches(batch_size=args.batch_size):
                validation_preds = np.concatenate((validation_preds, softmax(y_conv.eval(
                    feed_dict={x: validation_x, y_: validation_y, dropout: 0.0}), axis=1)))
                validation_labels = np.concatenate((validation_labels, y_.eval(
                    feed_dict={x: validation_x, y_: validation_y, dropout: 0.0})))
            validation_auc = roc_auc_score(validation_labels[1:], validation_preds[1:])

            debug('Validation AUC = %g' % validation_auc)

            # Save the best model
            if validation_auc > auc_tracker:
                saver.save(sess, os.path.join(args.save_dir, 'model.ckpt'))
                auc_tracker = validation_auc
                best_epoch = epoch

            # Learning rate schedule parameters
            warmup_steps = args.warmup_epochs * batches_per_epoch
            total_steps = args.epochs * batches_per_epoch

            # Train
            for _ in tqdm(range(batches_per_epoch)):

                # Update learning rate
                if global_step.eval() <= warmup_steps:
                    eta = args.init_lr + global_step.eval() * (args.max_lr - args.init_lr) / warmup_steps
                elif global_step.eval() <= total_steps:
                    eta = args.max_lr * (((args.final_lr / args.max_lr) ** (1 / (total_steps - warmup_steps))) ** (
                                global_step.eval() - warmup_steps))
                else:  # theoretically this case should never be reached since training should stop at total_steps
                    eta = args.final_lr

                # Training
                train_x, train_y = train_generator.next(batch_size=args.batch_size, data_aug=args.training_data_aug)
                train_step.run(feed_dict={x: train_x, y_: train_y, dropout: args.dropout, learning_rate: eta})
                if global_step.eval() % args.log_frequency == 0:
                    print('Train loss %g, lr %g' %
                          (loss.eval({x: train_x, y_: train_y, dropout: args.dropout, learning_rate: eta}), eta))

        # Evaluation on test set
        saver.restore(sess, os.path.join(args.save_dir, 'model.ckpt'))
        info('Best model validation AUC = %g on epoch %g' % (auc_tracker, best_epoch))
        test_total_accuracies = []
        test_preds = np.array([[0, 0]])
        test_labels = np.array([[0, 0]])
        for test_X, test_Y in test_generator.data_in_batches(batch_size=args.batch_size):
            test_total_accuracies.append(accuracy.eval(feed_dict={x: test_X, y_: test_Y, dropout: 0.0}))
            test_preds = np.concatenate((test_preds, softmax(
                y_conv.eval(feed_dict={x: test_X, y_: test_Y, dropout: 0.0}), axis=1)))
            test_labels = np.concatenate(
                (test_labels, y_.eval(feed_dict={x: test_X, y_: test_Y, dropout: 0.0})))

        test_accuracy = np.mean(test_total_accuracies)
        test_auc = roc_auc_score(test_labels[1:], test_preds[1:])

        info('Best model test AUC = %g, accuracy = %g' % (test_auc, test_accuracy))

    return test_auc, test_accuracy


# Run the program
if __name__ == '__main__':
    args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
    run_training(args, logger)

