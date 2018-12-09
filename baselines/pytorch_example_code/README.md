# 576nlp2018spring
class project repo for cse576 nlp

Revised March 14 to add sample code from pytorch_sentiment_rnn

This code implements several neural networks and associated data processing routines described in the paper, Tang et al., "Effective LSTMs for Target-Dependent Sentiment Classification," arXiv: 1512.01100v2 (Sept. 29, 2016).  It also makes use of pre-trained word embeddings discussed in Pennington et al., "GloVe: Global Vectors for Word Representation." (2014).

0. Dependencies include keras, numpy, tqdm, nltk, torch, torchvision, sklearn, and tensorflow.
This code recommends using a GPU, but it is not strictly necessary - I was able to run it with just the CPU version of tensorflow.  You should be able to install all of these using pip.  I recommend using a virtualenv.

1. Before running the code here, make a directory one level above
called "glove_embeddings" and put the following files in it:

   - glove.840B.300d.txt

   - glove_Restaurants_term.pkl

   The first (.txt) file above can be downloaded at this URL: https://nlp.stanford.edu/projects/glove/
   The second (.pkl) file is created if it doesn't already exist.  This creation might take a while, but you should only need to do it once.

   Before the next step, make sure to create the directory "store."  Check the file prepare.py at line 359 for more details.

2. python prepare.py

3. python train.py
   The above uses all defaults, which you can find by inspecting train.py.  
   If you wish to exert more control over the model, etc., you might try this command instead:

   python train.py --batch-size 20 --rnn_type GRU --lr 0.0001 --mdl TD-RNN --opt Adam
