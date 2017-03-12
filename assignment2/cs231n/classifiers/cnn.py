import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # The parameters of the conv is of size (F,C,HH,WW) with
    # F give the nb of filters, C,HH,WW characterize the size of
    # each filter
    # Input size : (N,C,H,W)
    # Output size : (N,F,Hc,Wc)
    C, H, W = input_dim
    F = num_filters
    HH = filter_size
    WW = filter_size
    stride = 1
    pad = (filter_size - 1)/2
    H_ = (H-HH + 2*pad)/stride + 1
    W_ = (W-WW + 2*pad)/stride + 1

    w1 = weight_scale * np.random.randn(F, C, HH, WW)
    b1 = np.zeros(F)

    # Pool layer : 2*2
    # The pool layer has no arameters but is important in the count of dimension.
    # Input : (N,C,H_, W_)
    # output: (N, F, Hp, Wp)
    width_pool = 2
    height_pool = 2
    stride_pool = 2
    Hp = (H_ - height_pool)/stride_pool + 1
    Wp = (W_ - width_pool)/stride_pool + 1


    # Hidden Affine layer
    # Input: (N, F*Hp*Wp)
    # Output: (N,Hh)
    Hh = hidden_dim
    w2 = weight_scale * np.random.randn(F*Hp*Wp, Hh)
    b2 = np.zeros(Hh)

    # Last output affine layer
    # Input: (N,Hh)
    # Output: (N,Hc)
    Hc = num_classes
    w3 = weight_scale * np.random.randn(Hh, Hc)
    b3 = np.zeros(Hc)

    self.params.update({'W1': w1, 'W2': w2, 'W3': w3, 'b1': b1, 'b2': b2, 'b3': b3})

    # pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_relu_pool_out, cache_conv_layer = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    affine_relu_out, cache_affine_relu_layer = affine_relu_forward(conv_relu_pool_out, W2, b2)
    scores, cache_affine_layer = affine_forward(affine_relu_out, W3, b3)

    # pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    loss = data_loss + reg_loss

    dx3,dw3,db3 = affine_backward(dscores, cache_affine_layer)
    dw3+=self.reg*W3

    dx2, dw2, db2 = affine_relu_backward(dx3, cache_affine_relu_layer)
    dw2 += self.reg*W2

    dx1,dw1,db1 = conv_relu_pool_backward(dx2, cache_conv_layer)
    dw1+=self.reg*W1

    grads.update({'W1': dw1,'b1': db1,'W2': dw2,'b2': db2,'W3': dw3,'b3': db3})
    
    # pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
# pass
