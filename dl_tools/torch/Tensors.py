import numpy as np

# N=>batch size, D_in=>input_dimension, H=>hidden dimension, D_out=> output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Get input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    
