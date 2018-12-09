import numpy as np
import theano
import argparse
import time
import sys
import json
import random
from tqdm import tqdm

from Optimizer import OptimizerList
from Evaluator import Evaluators
from DataManager import DataManager
from lstm_att_con import AttentionLstm as Model
from PrintError import printout

# Main training function
def train(model, train_data, optimizer, epoch_num, batch_size, batch_n):
    st_time = time.time()
    loss_sum = np.array([0.0, 0.0, 0.0])
    total_nodes = 0

    for batch in tqdm(range(batch_n), desc='Training progress for this epoch'):
        # Start and end are indexes used to slice training data into chunks (i.e. batches) to be sent to 'do_train' function
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(train_data))

        # Get loss values fof batch from 'do_train', add to running sum of loss values of whole set
        batch_loss, batch_total_nodes = do_train(model, train_data[start:end], optimizer)
        loss_sum += batch_loss
        total_nodes += batch_total_nodes # TODO, seems to be useless

    return loss_sum[0], loss_sum[2]
#end 'train'

# Workhorse training function, runs for each "batch" that is from total training data set
def do_train(model, train_data, optimizer):
    eps0 = 1e-8
    batch_loss = np.array([0.0, 0.0, 0.0])
    total_nodes = 0

    # TODO not sure what 'grad' here refers to, I'm guessing gradient?
    for _, grad in model.grad.items():
        grad.set_value(np.asarray(np.zeros_like(grad.get_value()), \
                dtype=theano.config.floatX))


    # Sends each item through theano's 'function' to determine loss values based on input and expected output
    for item in train_data:
        sequences, target, tar_scalar, solution =  item['seqs'], item['target'], item['target_index'], item['solution']
        batch_loss += np.array(model.func_train(sequences, tar_scalar, solution))
        total_nodes += len(solution)

    # TODO not sure what this is doing, I'm guessing normalizing the gradient based on data length?
    for _, grad in model.grad.items():
        grad.set_value(grad.get_value() / float(len(train_data)))

    # Update gradient values with optimizer function
    optimizer.iterate(model.grad)

    return batch_loss, total_nodes
# end 'do_train'

# Main testing function
def test(model, test_data, grained):

    # Initialization
    evaluator = Evaluators[grained]()
    keys = list(evaluator.keys())
    loss = .0
    total_nodes = 0
    correct = {key: np.array([0]) for key in keys}
    wrong = {key: np.array([0]) for key in keys}
    def cross(solution, pred):
        return -np.tensordot(solution, np.log(pred), axes=([0, 1], [0, 1]))

    # For each sentence in the test data, fetch the correct polarity and
    # determine loss value based on the model's predicted polarity
    for item in tqdm(test_data, desc='Testing progress for this epoch'):
        sequences, target, tar_scalar, solution =  item['seqs'], item['target'], item['target_index'], item['solution']
        pred = model.func_test(sequences, tar_scalar)
        loss += cross(solution, pred)
        total_nodes += len(solution)
        result = evaluator.accumulate(solution[-1:], pred[-1:]) # TODO, seems to be useless

    # Accuracy statistic for both three-way and binary classification
    acc = evaluator.statistic()

    return loss/total_nodes, acc
# end 'test'


# Used to run test_data through model and print error cases
def test1(dataset, datasplit, model, test_data, grained):
    evaluator = Evaluators[grained]()
    keys = list(evaluator.keys())
    def cross(solution, pred):
        return -np.tensordot(solution, np.log(pred), axes=([0, 1], [0, 1]))

    error_index = [];
    error_pred = [];
    i = 0
    for item in tqdm(test_data, desc='Final test'):
        sequences, target, tar_scalar, solution =  item['seqs'], item['target'], item['target_index'], item['solution']
        pred = model.func_test(sequences, tar_scalar)
        pred[np.where(pred == np.max(pred))] = 1
        pred[np.where(pred < 1)] = 0
        if np.tensordot(solution, pred, axes = ([0,1], [0,1])) == 0:
            error_index.append(i)
            n = np.argmax(pred, axis=1)
            if n == 0:
                e = -1
            elif n == 1:
                e = 0
            elif n == 2:
                e = 1
            error_pred.append(e)
        i = i + 1
    printout(dataset, datasplit, error_index, error_pred)
# end 'test1'

###############################
### BEGIN MAIN PROGRAM BODY  ##
###############################

if __name__ == '__main__':
    argv = sys.argv[1:]                                                     # Slice off the first element of argv (which would just be the name of the program)


    ################################################
    ##  BEGIN SETTING UP DEFAULT HYPERPARAMETERS  ##
    ################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lstm')                 # Name will be written to the results file
    parser.add_argument('--seed', type=int, default=int(1000*time.time()))  # Seed used for 'random'  module, which will shuffle training data
    parser.add_argument('--dim_hidden', type=int, default=300)              #
    parser.add_argument('--dim_gram', type=int, default=1)                  #
    parser.add_argument('--dataset', type=str, default='data')              # Name will be used to reference folder to find data files in (i.e. data files are in ./data)
    parser.add_argument('--fast', type=int, choices=[0, 1], default=0)      #
    parser.add_argument('--screen', type=int, choices=[0, 1], default=0)    #
    parser.add_argument('--optimizer', type=str, default='ADAGRAD')         # Specifies that 'adagrad' gradient descent algorithm will be used for parameter learning
    parser.add_argument('--grained', type=int, default=3)                   # TODO Not sure what this is for, but it is referenced a lot
    parser.add_argument('--lr', type=float, default=0.01)                   # General learning rate to be used for optimizer?
    parser.add_argument('--lr_word_vector', type=float, default=0.1)        # Learning rate of word vector to be used for optimizer?
    parser.add_argument('--epoch', type=int, default=25)                    # Number of epochs to use during the train/testing process
    parser.add_argument('--batch', type=int, default=25)                    # Size of data batches when doing training
    #############################################
    ## END SETTING UP DEFAULT HYPERPARAMETERS  ##
    #############################################


    args, _ = parser.parse_known_args(argv)                                 # Overwrite default hyperparameters if specified in command-line call

    random.seed(args.seed)                                                  # Will be used to shuffle training data
    print("### MAIN.PY: Initializing data set ...")
    data = DataManager(args.dataset)                                        # New instance of DataManage obect, defined in DataManager.py
    print("### MAIN.PY: Preparing list of words from dictionary...")
    wordlist = data.gen_word()                                              # Get comprehensive list of words from data
    print("### MAIN.PY: Formatting data...")
    train_data, dev_data, test_data = data.gen_data(args.grained)           # Store formatted data retrieved from *.cor files
    print("### MAIN.PY: Initializing model...")
    model = Model(wordlist, argv, len(data.dict_target))                    # Initialize new model (specified in lstm_att_con.py in this case)
    batch_n = int((len(train_data) - 1) / args.batch + 1)                   # Determine the number of batches to split training data into
    print("### MAIN.PY: Initializing optimizer...")
    optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_word_vector) # Intialize instance of OptimizerList (defined as ADAGRAD object in Optimizer.py)
    details = {'loss': [], 'loss_train':[], 'loss_dev':[], 'loss_test':[], \
            'acc_train':[], 'acc_dev':[], 'acc_test':[], 'loss_l2':[]}      # Initiaize sections of result data that will be printed to final results file

    ##########################################
    ##  BEGIN TRAINING AND TESTING ON DATA  ##
    ##########################################
    print("### MAIN.PY: Starting training and testing ...")
    # For the number of epochs defined, performed training and testing process
    for e in tqdm(range(args.epoch), desc='Epoch progress'):

        # Shuffle training data to avoid completely deterministic results
        random.shuffle(train_data)

        # Will hold loss data during training and testing for this epoch
        now = {}

        # Train
        now['loss'], now['loss_l2'] = train(model, train_data, optimizer, e, args.batch, batch_n)

        # Test on data set from train.cor, dev.cor, and test.cor
        now['loss_train'], now['acc_train'] = test(model, train_data, args.grained)
        now['loss_dev'], now['acc_dev'] = test(model, dev_data, args.grained)
        now['loss_test'], now['acc_test'] = test(model, test_data, args.grained)

        # Write the loss data gathered during training and testing to the appropriate sections of
        # the details dictionary, which will be written to final restults file
        for key, value in list(now.items()):
            details[key].append(value)
        with open('result/%s.txt' % args.name, 'w+') as f:
            f.writelines(json.dumps(details))

    # Do final run through test data and print error cases
    # test1(model, test_data, args.grained)
    # revised to tailor to dataset
    test1(args.dataset, 'test', model, test_data, args.grained)

    ########################################
    ##  END TRAINING AND TESTING ON DATA  ##
    ########################################

    # END OF FILE
