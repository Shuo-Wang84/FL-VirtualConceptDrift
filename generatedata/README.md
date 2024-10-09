# Synthetic Scenario Design:
To generate data streams for FL, we fix the number of clients to 10. 
For each client, we generate 100 data points per batch. 
For the Gaussian (2D), Gaussian (8D) and Hyperplane datasets, each client receives 200 batches (i.e. time steps), leading to 20,000 data samples in total. 
Except for the asynchronous and Correlation feature data, all other datasets experience a drift at time steps = 100, 
with each case experiencing a drift event at batch 100 (the midpoint of the data stream). 

Specifically, for the Gaussian (2D) and Gaussian (8D) data,
the data is generated using standard normal distribution Gaussian functions before the drift, and post-drift, the data is linearly transformed using a randomly generated transformation matrix. 
For the Hyperplane dataset, we use the ``HyperplaneGenerator" object to generate the data stream, dividing the data into two concepts before and after the drift based on the decision boundary $y = x$.

We simulated 11 virtual drift scenarios, as shown in Table \ref{tab:scenario}, altering four spatial characteristics (synchronism, coverage, direction, correlation). 
These four characteristics are designed as follows: Asyn/Syn indicates whether the virtual drift time is consistent across 10 clients. 
Syn means Synchronism, and Asyn means asynchronism. Coverage denotes the proportion of clients experiencing virtual drift. 
Direction describes whether the drift direction is the same among clients. 
Correlation indicates whether there is inter-client correlation among drifting clients.
