import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] 
  #num_classes是种类的个数
  num_train = X.shape[0] 
  #num_train是分类目标的个数
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W) 
    #X.shape (500,3073) X[i].shape (3073,) scores.shape (3073,)
    correct_class_score = scores[y[i]]
    #把分类器分类的正确得分保存到 correct_class_score 中
    for j in range(num_classes):
      if j == y[i]:    #计算margin且不把自身带入计算
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] += -X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  num_classes = W.shape[1]
  score = X.dot(W)
  correct_class_score = score[range(num_train),y].reshape(-1,1)
  margin = np.maximum(0 , score - correct_class_score + 1)
  margin[range(num_train),y] = 0
  loss = np.sum(margin) / num_train + reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margin > 0] = 1
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
  #标记出现 margin > 0 的矩阵位置。通过数学转换得知对应于X_i和X_j出现的位置和次数
  dW = (X.T).dot(coeff_mat)
  dW = dW / num_train + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
