import numpy as np
from random import shuffle
# from past.builtins import xrange

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
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D = X.shape
  C = W.shape[1]

  for n in range(N):
    Xi = X[n,:]           # [1xD]
    score_i = Xi.dot(W)   # [1xC]
    max_score_i = -score_i.max()
    exp_score_i = np.exp(score_i + max_score_i)
    exp_score_total_i = np.sum(exp_score_i)

    numer = np.exp(score_i[y[n]] + max_score_i)
    denom = exp_score_total_i
    loss += -np.log(numer/denom)

    for c in range(C):
      if c==y[n]:
        dW[:, c] += -Xi + (exp_score_i[c]/exp_score_total_i) * Xi
      else:
        dW[:, c] += (exp_score_i[c] / exp_score_total_i) * Xi

  loss = loss / float(N) + 0.5 * reg * np.sum(W*W)
  dW = dW / N + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = W.shape[1]

  score = X.dot(W)                # [NxC]
  max_score = -score.max(axis=1)  # [1xC]
  exp_score = np.exp(score + max_score)
  exp_score_total = np.sum(exp_score, axis=1) #[1xC]

  numer = score[:]
  for n in range(N):
    Xi = X[n,:]           # [1xD]
    score_i = Xi.dot(W)   # [1xC]
    max_score_i = -score_i.max()
    exp_score_i = np.exp(score_i + max_score_i)
    exp_score_total_i = np.sum(exp_score_i)

    numer = np.exp(score_i[y[n]] + max_score_i)
    denom = exp_score_total_i
    loss += -np.log(numer/denom)

    for c in range(C):
      if c==y[n]:
        dW[:, c] += -Xi + (exp_score_i[c]/exp_score_total_i) * Xi
      else:
        dW[:, c] += (exp_score_i[c] / exp_score_total_i) * Xi

  loss = loss / float(N) + 0.5 * reg * np.sum(W*W)
  dW = dW / N + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

