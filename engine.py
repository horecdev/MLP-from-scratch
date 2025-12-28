import numpy as np
import numpy.typing as npt
 
Tensor = npt.NDArray[np.float64] # So we get type hints

class Linear:
    def __init__(self, input_size, output_size):
        # Initialize parameters
        self.weights: Tensor = np.random.randn(input_size, output_size)
        self.bias: Tensor = np.zeros((1, output_size))
        # Gradients (to be updatede)
        self.dW: Tensor | None = None
        self.db: Tensor | None = None
        
        self.input_cache: Tensor | None = None
        
        def forward(self, x: Tensor) -> Tensor:
            # x is of shape (B, input_size)
            # returns (B, output_size)
            self.input_cache = x # Saves what was inputted to then do backward pass
            return x @ self.weights + self.bias
        
        def backward(self, grad_output: Tensor) -> Tensor:
            # grad_output has to be of shape (B, output_size)
            
            self.dW = self.input_cache.T @ grad_output # (input_size, output_size)
            
            self.db = np.sum(grad_output, axis=0, keepdims=True) # (1, output_size)
            
            grad_input = grad_output @ self.weights.T # (B, input_size)
            
            return grad_input
        
class ReLU:
    def __init__(self):
        self.input_cache: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        self.input_cache = x
        return np.maximum(0, x) # Same shape as x
    
    def backward(self, grad_output: Tensor) -> Tensor:
        relu_grad = (self.input_cache > 0).astype(float) # If input wasn't clipped then it influences output by factor of 1
        return relu_grad * grad_output # Element-wise multiplication (same shapes)
    
class SoftmaxCrossEntropy: # e^x explodes quickly, so we implement softmax and loss at once to use a shortcut
    # If we need softmax during inference we just write a short softmax() func, we dont really need anything else.
    def __init__(self):
        self.logits: Tensor | None = None
        self.targets: Tensor | None = None
        self.probs: Tensor | None = None # This is the softmax output
    def forward(self, logits: Tensor, targets: Tensor) -> float:
        self.logits = logits
        self.targets = targets
        # First we offset the logits so that max value is 0 -> prevents np.exp() from totally exploding into inf. 
        # Math in this case is exactly the same
        max_logits = np.max(logits, axis=1, keepdims=True) # (B, 1)
        shifted_logits = logits - max_logits # (B, output_size)
        
        
    
class MLP:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
        
    def step(self, learning_rate):
        # Manual SGD
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= learning_rate * layer.dW
                layer.bias -= learning_rate * layer.db
                
                