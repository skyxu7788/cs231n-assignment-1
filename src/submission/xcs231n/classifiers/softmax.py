from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    #   y[i] is the column index of the correct class for row i (example i).
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        # shape (C)
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        # ### START CODE HERE ###
        for j in range(num_classes):
            dW[:,j] += p[j] * X[i]
            # (D,) += (C) * (D,), X[i] is a row vector of lenth D (one example's features)
        # ### END CODE HERE ###

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################num###############################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # ### START CODE HERE ###

    dW += 2*reg*W

            
    # ### END CODE HERE ###

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # - W: A numpy array of shape (D, C) containing weights.
    # - X: A numpy array of shape (N, D) containing a minibatch of data.
    # - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # ### START CODE HERE ###
    # (N,C)
    scores = X @ W 
    scores -= np.max(scores, axis=1, keepdims=True)
    N = X.shape[0]
    p = np.exp(scores)
    p /= np.sum(p, axis=1, keepdims=True)
    # get the score in row x (training data's example x) that matches with y[x](correct label for the xth example of data)
    #  for each row, find the element(score) that matches the y, y is an array of indexes for label
# p[arrX, arrY], arrX is the row we want to use arrY to index into, hence shape of p[arrX, arrY] is num of rows in the matix being performed indexing
    score_correct_label = p[np.arange(N),y]
    # print('shape of feature correct label %f', score_correct_label.shape)
    loss = -np.sum(np.log(score_correct_label))/N
    loss += reg * np.sum(W*W)
   
    # ### END CODE HERE ###

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # ### START CODE HERE ###
    # derivatives of loss with respoect to each class score
    dScores = p.copy()
    # one hot target: correct class probability should be 1, otehr is 0 
    # dScores = p - target, only correct class score gradient in each example -1
    dScores[np.arange(N), y] -= 1
    dScores /= N
    dW = np.transpose(X) @ dScores
    dW += 2*reg*W
    # ### END CODE HERE ###

    return loss, dW
