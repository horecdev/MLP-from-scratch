import numpy as np
import numpy.typing as npt
 
Tensor = npt.NDArray[np.float64] # So we get type hints

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weights = np.random.randn(num_embeddings, embedding_dim)
        self.grad: Tensor | None = None
        self.input_cache: Tensor | None = None
    
    def forward(self, indices: Tensor): # Indices of shape (B, seq_len)
        self.batch_size = indices.shape[0]
        self.input_cache = indices
        
        embeds = self.weights[indices] # (B, seq_len, embedding_dim)
        return embeds.reshape(self.batch_size, -1) # (B, seq_len * embedding_dim)
    
    def backward(self, grad_output: Tensor) -> None:
        self.dW = np.zeros_like(self.weights)
        
        np.add.at(self.dW, self.input_cache, grad_output)
        # Iterates through input_cache and grad_output in parallel
        # Looks at index in input_cache[a, b], takes vector at grad_output[a, b]
        # and adds it in self.dW in place indicated by index
        # The index was "which one of the vectors in weights did I take?"
        # The vector at grad_output is "I took this vector from embeds, how does it influence the loss?"
        # If you add how it influences the loss to the embedding vector of the item you took,
        # and you accumulate then you have full gradient.
        # We dont return anything because there is nothing to return. Embedding is the first step.
        
class Flatten:
    def __init__(self):
        self.input_shape = None
        
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
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
    
class SoftmaxCrossEntropy: # in backward implementation we use a shortcut, which if we implemented them separately
    # would make us potentially do math like tiny number * huge number what is pointless and prone to errors (it happens in chain rule)
    # If we need softmax during inference we just write a short softmax() func, we dont really need anything else.
    # Softmax is for single class output problems as the probs sum to 1, we cant have two correct classes
    def __init__(self):
        self.logits: Tensor | None = None
        self.targets: Tensor | None = None
        self.probs: Tensor | None = None # This is the softmax output
        
    def forward(self, logits: Tensor, targets: Tensor) -> float:
        self.logits = logits # (B, output_size)
        self.targets = targets # (B, output_size) 
        # First we offset the logits so that max value is 0 -> prevents np.exp() from totally exploding into inf. 
        # Math in this case is exactly the same
        max_logits = np.max(logits, axis=1, keepdims=True) # (B, 1)
        shifted_logits = logits - max_logits # (B, output_size)
        # We shift because softmax is translation-invariant (explanation below return statement in the func)
        
        exp_logits = np.exp(shifted_logits) # (B, output_size)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True) # (B, 1)
        self.probs = exp_logits / sum_exp # (B, output_size)
        
        # cross-entropy loss is -log of probability of correct class, lets assume ours is one-hot encoded 
        # and we will apply for it later in sampling / training
        
        log_probs = np.log(self.probs + 1e-15) # (B, output_size), we also prevent the log(0)
        log_probs = log_probs * targets # Eliminates all other classes that weren't one-hot encoded
        batch_loss = -np.sum(log_probs, axis=1) # (B)
        batch_loss = np.mean(batch_loss, axis=0)
        
        return batch_loss
    
        # some intuition: prob is (log(e^x_i / sum(e^x_j))
        # so prob is x_i - log(sum(e^x_j))
        # Lets look first at softmax(x_i) = (e^x_i - sum(e^x_j))
        # softmax(x_i - max_logit) where max_logit = c = e^(x_i - c) / sum(e^(x_j - c)) - this means each logit is offset
        # = (e^x_i * e^-c / sum(e^x_j * e^-c)) because the e^-c is constant term and appears in every term 
        # on the bottom we can factor it out in front of the sum
        # = (e^x_i * e^-c / e^-c * sum(e^x_j)
        # the e^-c cancels
        # = e^x_i / sum(e^x_j) we get what we started with
    
    def backward(self) -> Tensor:
        # Calculate gradient wrt. logits
        # gradient derivation of softmax + cross entropy simplifies to 
        # dL/dLogits = (Probs - Trues / batch_size)
        # the batch_size is because we take the mean, so intuitively weach prob contributes just 1/N
        # I derived it on paper and it checks
        batch_size = self.probs.shape[0]
        
        grad = (self.probs - self.targets) / batch_size
        
        return grad
        
        # Intuitively: 
        # if target = 1 (correct class) the loss is prob - 1, so only for a model that is sure it has no effect
        # if prob < 1 then grad < 0 so increasing the logit decreases the loss
        # if target = 0 (other class) then the loss is the prob, which is positive as long as prob != 0
        
        
    
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
            if hasattr(layer, 'weights') and layer.dW is not None:
                layer.weights -= learning_rate * layer.dW
            if hasattr(layer, 'bias') and layer.db is not None:
                layer.bias -= learning_rate * layer.db
                
                