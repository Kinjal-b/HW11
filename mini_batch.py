# This implements a mini-batch approach. 
# It generates a specified number of mini-batches from the input dataset, each containing a subset of the data with the designated batch size

import numpy as np

def create_mini_batches(X, y, batch_size):
    """
    Creates a list of mini-batches from the dataset.
    
    Parameters:
    X (numpy.ndarray): Input features matrix.
    y (numpy.ndarray): Output labels vector.
    batch_size (int): Size of each mini-batch.
    
    Returns:
    list of tuples: Each tuple contains a mini-batch of input features and output labels.
    """
    mini_batches = []
    data = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    
    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, y_mini))
    
    if data.shape[0] % batch_size != 0:
        mini_batch = data[n_minibatches * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, y_mini))
        
    return mini_batches

# Example usage
if __name__ == "__main__":
    # Generate some random data
    X = np.random.randn(1024, 10)  # 1024 samples, 10 features
    y = np.random.randn(1024, 1)   # 1024 target values
    
    batch_size = 64  # Example batch size
    mini_batches = create_mini_batches(X, y, batch_size)
    
    print(f"Total mini-batches created: {len(mini_batches)}")
    print(f"First mini-batch feature shape: {mini_batches[0][0].shape}")
    print(f"First mini-batch label shape: {mini_batches[0][1].shape}")

