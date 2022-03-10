import time

import torch
from torch import nn


# 1a)
def polynomial_fun(x: float, w: torch.Tensor):
    """
    Returns the 3rd order polynomial using some input weights (w)
	Parameters:
		x (float): A floating integer
		w (list): Mx1 vector of weights, (M being poly. order)
	Returns:
		y (float): Output of polynomial
    """
    y = 0
    M = w.shape[0] - 1
    for order in range(M, -1, -1):
        y += (x**order) * w[M-order, :] # (x**3)*w[0,:] + (x**2)*w[1,:] + x*w[2,:] + w[3,:]
    return y


# 1b)
def fit_polynomial_ls(data: torch.Tensor, M: int) -> torch.Tensor:
    """
    Uses least squares regression to fit a M order polynomial to data
	Parameters:
		data (torch.Tensor): A 2xN tensor of x and y values
		M (int): Order of polynomial to be fit to data
	Returns:
		w (torch.Tensor): (M+1)x1 tensor of weights W
    """
    N = data.shape[0]
    print(f"Fitting {N} data points to a {M} order polynomial")

    x = data[:,0]
    y = data[:,1].reshape(N,1)

    X = torch.vander(x, M+1)
    w = (X.T @ X).inverse() @ (X.T @ y)
    return w


# 1c)
def fit_polynomial_sgd(data: torch.Tensor, M: int, learning_rate: float, batch_size: int) -> torch.Tensor:
    """
    Minibatch SGD Algorithim using MSE for loss
	Parameters:
		data (torch.Tensor): A 2xN matrix of x and y values
		M (int): Order of polynomial to be fit to data
		learning_rate (float): A 2xN matrix of x and y values
		batch_size (int): A 2xN matrix of x and y values
	Returns:
		w (torch.Tensor): (M+1)x1 tensor of weights W
    """
    def meanSqrError(y, yHat):
        return torch.mean((y - yHat)**2)

    def randomiseAndBatch(_data):
        N = _data.size(dim=0)
        idx = torch.randperm(N)
        x, y = _data[:,0][idx], _data[:,1][idx]
        for i in range(0, N, batch_size):
            yield x[i:i+batch_size], y[i:i+batch_size]

    w = torch.randn((M+1,1))
    w = nn.Parameter(w)

    totalEpochs = 50
    for epoch in range(1, totalEpochs+1):
        print(f"=== Epoch {epoch} ===")
        
        # Shuffle and batch data
        dataIter = randomiseAndBatch(data)
        for batch, (x, y) in enumerate(dataIter):
            
            # Predict and compute loss
            yHat = torch.vander(x, M+1) @ w
            loss = meanSqrError(y, yHat.flatten())
            loss.backward()

            # Update weights
            with torch.no_grad():
                w -= (w.grad * learning_rate)
                w.grad.zero_()

            # Periodically report loss
            if (batch+1) % 5 == 0:
                print("Minibatch {}: MSE of {:.3f}".format(batch+1, loss))

    print("=== Final Weights ===")
    print(w.data)
    return w.data


# 1d.VI) Accuracy of w and y using R-squared measure
def get_accuracy(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Calcualtes accuracy of a prediciton given some ground truth"""
    gtMean = torch.mean(ground_truth).item()
    return 1 - sum((ground_truth - predicted)**2)/sum((ground_truth-gtMean)**2)


# 1d.VI) RMSE of w and y
def root_mean_square(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Calcualtes root mean square of a prediciton given some ground truth"""
    return torch.sqrt(torch.mean((predicted - ground_truth)**2)).item()


if __name__ == "__main__":
    print("\n#### COMP0090-CW1 - Task 1 ####\n")
    # 1d.I) Intialise M = 3 and w = [1,2,3,4].T
    M = 3
    w = torch.tensor([[1., 2., 3., 4.]]).T
    
    # 1d.I) Generate training set with Gauss noise
    x_train = torch.randint(-20, 20, (100,), dtype=torch.float32)
    y_train_true = x_train.clone().apply_(lambda x: polynomial_fun(x, w))
    y_train = torch.normal(y_train_true, 0.2) # Add guassian noise sd=0.2
    train_data = torch.stack((x_train, y_train), dim=1)

    # 1d.I) Generate test set with Gauss noise
    x_test = torch.randint(-20, 20, (50,), dtype=torch.float32)
    y_test_true = x_test.clone().apply_(lambda x: polynomial_fun(x, w))
    y_test = torch.normal(y_test_true, 0.2) # Add guassian noise sd=0.2
    test_data = torch.stack((x_test, y_test), dim=1)

    # 1d.II) Find w using fit_polynomial_ls() and compute y_hat_ls
    M_ls = 4
    start_ls = time.time()
    w_ls = fit_polynomial_ls(train_data, M_ls)
    end_ls = time.time()
    y_hat_ls_train = (torch.vander(x_train, M_ls+1) @ w_ls).flatten()
    y_hat_ls_test = (torch.vander(x_test, M_ls+1) @ w_ls).flatten()

    # 1d.III) Mean/sd differences between train data and true polynomial
    y_train_delta = y_train - y_train_true
    y_train_delta_mean = torch.mean(y_train_delta)
    y_train_delta_std = torch.std(y_train_delta)
    print("\nTrain set vs True Polynomial - Mean: {:.3f}, Std: {:.3f}".format(y_train_delta_mean, y_train_delta_std))

    # 1d.III) Mean/sd differences between y_hat_ls and true polynomial
    y_hat_ls_train_delta = y_hat_ls_train - y_train_true
    y_hat_ls_train_delta_mean = torch.mean(y_hat_ls_train_delta)
    y_hat_ls_train_delta_std = torch.std(y_hat_ls_train_delta)
    print("\nLS predicted vs True Polynomial - Mean: {:.3f}, Std: {:.3f}".format(y_hat_ls_train_delta_mean, y_hat_ls_train_delta_std))

    # 1d.IV) Find w using fit_polynomial_sgd() and compute y_hat_sgd
    M_sgd = 4
    learning_rate, batch_size = 1e-10, 10 # Found generally good perf. with these + 50 epochs
    start_sgd = time.time()
    print("\nRunning Minibatch SGD...")
    w_sgd = fit_polynomial_sgd(train_data, M_sgd, learning_rate, batch_size)
    end_sgd = time.time()
    y_hat_sgd_train = (torch.vander(x_train, M_sgd+1) @ w_sgd).flatten() 
    y_hat_sgd_test = (torch.vander(x_test, M_sgd+1) @ w_sgd).flatten()

    # 1d.V) Mean/sd differences between y_hat_sgd and true polynomial
    y_hat_sgd_train_delta = y_hat_sgd_train - y_train_true
    y_hat_sgd_train_delta_mean = torch.mean(y_hat_sgd_train_delta)
    y_hat_sgd_train_delta_std = torch.std(y_hat_sgd_train_delta)
    print("\nSGD predicted vs True Polynomial - Mean: {:.3f}, Std: {:.3f}".format(y_hat_sgd_train_delta_mean, y_hat_sgd_train_delta_std))

    # Scale 3rd order weights to 4th order for RMSE calcualations
    w_scaled = torch.cat([w, torch.zeros((1,1))], dim=0)

    # 1d.VI) Accuracy of fit_polynomial_ls() vs. test set and RMSE of w and y
    y_hat_ls_test = (torch.vander(x_test, M_ls+1) @ w_ls).flatten()
    test_ls_acc = get_accuracy(y_hat_ls_test, y_test_true) # R-squared
    w_ls_rmse = root_mean_square(w_ls, w_scaled)
    y_ls_rmse = root_mean_square(y_hat_ls_test, y_test_true)
    print("\n=== LS Test Accuracy ===")
    print("LS test set accuracy using R-squared: {:.3f}".format(test_ls_acc))
    print("w LS RMSE: {:.3f}".format(w_ls_rmse))
    print("y LS RMSE: {:.3f}".format(y_ls_rmse))

    # 1d.VI) Accuracy of fit_polynomial_sgd() vs. test set and RMSE of w and y
    y_hat_sgd_test = (torch.vander(x_test, M_sgd+1) @ w_sgd).flatten()
    test_sgd_acc = get_accuracy(y_hat_sgd_test, y_test_true) # R-squared
    w_sgd_rmse = root_mean_square(w_sgd, w_scaled)
    y_sgd_rmse = root_mean_square(y_hat_sgd_test, y_test_true)
    print("\n=== SGD Test Accuracy ===")
    print("SGD test set accuracy using R-squared: {:.3f}".format(test_sgd_acc))
    print("w SGD RMSE: {:.3f}".format(w_sgd_rmse))
    print("y SGD RMSE: {:.3f}".format(y_sgd_rmse))

    # 1d.VII) Compare speed of fit_polynomial_ls() vs. fit_polynomial_sgd()
    ls_time = round(end_ls - start_ls, 5)
    sgd_time = round(end_sgd - start_sgd, 5)
    print("\n=== LS vs. SGD Speed ===")
    print("Least Squares total time: {:.2f}s".format(ls_time))
    print("Minibatch SGD total time: {:.2f}s".format(sgd_time))
    print("Time Difference: {:.2f}s".format(sgd_time-ls_time))
