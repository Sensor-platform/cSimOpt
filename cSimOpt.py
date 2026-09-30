# Copyright © UIF (University Industry Foundation), Yonsei University. All rights reserved. For any inquiries regarding the usage or licensing of this code, please contact the UIF. (Contact: +82-2-2123-5176 / jemin17@yonsei.ac.kr)

import os
import numpy as np
import datetime
import uuid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import tensorflow as tf

from scipy.stats import norm

from dpl_surrogate import DPL # DPL surrogate model

from itertools import product

import random

import copy


def sensor_parameter_decoder(step_size, start_point, num_points): # Re-scaling sensor parameters
    """
    Decoding sensor parameters into real values
    """
    start_point_map = {1: 32, 2: 36, 3: 40, 4: 44, 5: 48}
    start_point = start_point_map.get(start_point, start_point)

    if step_size == 1:
        step_size = 1
        num_points = int(24 + ((num_points - 1) * 8))
    elif step_size == 2:
        step_size = 2
        num_points = int(12 + ((num_points - 1) * 4))
    elif step_size == 3:
        step_size = 4
        num_points = int(6 + ((num_points - 1) * 2))

    return step_size, start_point, num_points


def layer_decoder(layer): # Re-scaling # of units in hidden layers
    """ Decoder for # of units in hidden layers
    Args:
        layer: int
            Embedding # of units in hidden layers
    """
    return 2 * layer


def learning_rate_decoder(lr): # Re-scaling learning rate
    """ Decoder for learning rate
    Args:
        lr: int
            Embedding learning rate
    """
    return 0.0002*lr+0.0003

def objective_function(x_idx, cost, sensor_scores): # Load the prediction accuracy at each configuration and dataset size
    return sensor_scores[x_idx, int(cost / 1000)-1]

dims = [3,5,4,5,5,6] # Dimension of each configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENSOR_SCORES_PATH = os.path.join(BASE_DIR, "scores_performance.npy")
sensor_scores = np.load(SENSOR_SCORES_PATH)

# Define a surrogate model as a DPL
class DPLEnsemble:
    def __init__(self, num_features=6, num_models=5, seed=0):
        """
        Args:
            num_features: int
                Number of input features of a DPL model.
            num_models: int
                Number of DPL models for a surrogate model.
            seed: int
                Seed.
        """
        self.num_features = num_features
        self.num_models = num_models
        self.seed = seed

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def fit(self, X, Y, max_cost=5000, learning_rate=0.001, num_epochs=50):
        """ Train the surrogate model for observed performances.
        Args:
            X: np.ndarray
                Input. (num_samples,num_features+1)
            Y: np.ndarray
                Performane. (num_samples,)
        """
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        inputs = np.hstack((X[:,:-1].reshape(-1,6), X[:,-1:]/max_cost)) # Configuration + normalized cost
        self.models = []
        for _ in range(self.num_models):
            model = DPL()
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=tf.keras.losses.MeanAbsoluteError()) # Adam optimizer, MAE loss
            model.build(input_shape=(1, self.num_features+1)) # parameters and cost
            model.fit(x=inputs, y=Y, epochs=num_epochs, verbose=0) # Train DPL
            self.models.append(model)
    
    def predict(self, x, cost=1):
        """ Predict the performance of a given input at a given cost.
        Args:
            x: np.ndarray
                Parameter configurations with size (6,).
            cost: float
                Normalized cost to predict the performance.
        Returns:
            mu: float
                Mean of the performance.
            sigma: float
                Standard deviation of the performance.
        """
        x = x.reshape(-1,6)
        inputs = np.hstack((x, np.full((len(x),1), cost))).astype(np.float32) # Configuration + dataset size
        all_predictions = [model.predict(inputs, verbose=0) for model in self.models] # Predictions of all neural networks in an ensemble

        return np.mean(all_predictions, axis=0), np.std(all_predictions, axis=0) # Mean and std prediction

class BayesianOptimizerMF:
    def __init__(self, bounds, surrogate='gp', acq='ucb', n_initial_points=1, kappa=2.5, max_cost=5000, unit_cost=1000, seed=0):
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.surrogate_model = DPLEnsemble(len(self.bounds), seed = seed) # DPL surrogate model
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4) # GP surrogate model
        self.X_sample = np.empty((0, len(self.bounds)+1)) # (num_samples, num_features+1) The last feature represents a cost
        self.Y_sample = np.array([])

        self.surrogate = surrogate
        self.acq = acq

        self.kappa = kappa
        
        self.max_cost = max_cost # Maximum amount of collected data for each sensor parameter configuration
        self.unit_cost = unit_cost # Unit amount of collected data
        self.cost = dict() # Amount of data for performance evaluation for each parameter configuration
        self.cost_sensor = dict() # Amount of collected data for each 'sensor' parameter configuration
        
        self.best_value = 0 # Best observed performance to calculate EI
        self.tau = 0.9658 # Terminal condition
        
        self.seed = seed # Fix seeds

        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.total_collected_data = 0 # Total amount of collected data
        self.total_evaluated_data = 0 # Total amount of data for performance evauation
        self.total_collected_potential_data = 0 # Total amount of collected potential data
        self.measurement = list() # Measurements for save

        self.count = dict() # Total count of selected points

        self.sp_1 = [1,2,3]
        self.sp_2 = [1,2,3,4,5]
        self.sp_3 = [1,2,3,4]
        self.hp_1 = [1,2,3,4,5]
        self.hp_2 = [1,2,3,4,5]
        self.hp_3 = [1,2,3,4,5,6]
        self.candidates = np.array([list(p) for p in product(self.sp_1, self.sp_2, self.sp_3, self.hp_1, self.hp_2, self.hp_3)])
    
    def _acquisition(self, x):
        """ Expected improvement at fidelity level of maximum cost (self.max_cost)
        Arg:
            x: np.ndarray
                Parameter configurations.
        """

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        best_value = np.max(-self.Y_sample) # Best observed value
        # Mean and std of the performance of x at c_max
        if self.surrogate == 'gp':
            mu, sigma = self.gp.predict(np.hstack([x, np.ones((len(x), 1))]), return_std=True)
        elif self.surrogate == 'dpl':
            mu, sigma = self.surrogate_model.predict(x, 1)

        if self.acq == 'ei':
            imp = -mu - best_value # Negated mean is the real performance
            z = imp / sigma
            return imp * norm.cdf(z) + sigma * norm.pdf(z) # Expected improvement
        elif self.acq == 'ucb':
            return self.kappa * sigma - mu # Upper confidence bound


    def optimize(self, total_amount=120000):
        """
        Execute Bayesian Optimization loop
        """
        ## Fix seeds
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        ## Fix seeds

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_day_of_week = datetime.datetime.now().strftime("%A")
        tag = str(uuid.uuid4())
        
        experiment_name = f"Bayesian_Optimization_{current_date}_{current_day_of_week}_{tag}"
        print(f"[Start Experiment] {experiment_name}")
        
        # Random points
        param_lists = [self.sp_1, self.sp_2, self.sp_3, self.hp_1, self.hp_2, self.hp_3]
        
        # Sample n_initial_points unique configurations without replacement
        results = random.Random(self.seed).sample(list(product(*param_lists)), self.n_initial_points)
        # Random points

        iterations = 0
        current_objective = 0.1
        current_objective_max = 0.1
        while (current_objective <= self.tau) or (current_objective_max != current_objective):
            if iterations < self.n_initial_points:
                x = np.array(results[iterations]) # Random points
            else:
                exclude = [k for k, v in self.count.items() if v == int(self.max_cost / self.unit_cost)] # Exclude the configuration whose ML performance is evaluated at the maximum budget
                exclude = np.array([list(t) for t in exclude])
                self.candidates = self.candidates[~np.any([np.all(self.candidates == b, axis=1) for b in exclude], axis=0)]
                self.candidates = self.candidates.reshape(-1,6)
                x = self.candidates[np.argmin(- self._acquisition(self.candidates))] # Select next configuration

            key, s_key = tuple(x), tuple(x[:3])

            # Update the cost (amount of data) for the selected sensor parameter and hyperparameter configurations
            if s_key not in self.cost_sensor: # Case when b(theta_s)==0
                self.cost[key] = self.cost_sensor[s_key] = self.unit_cost
                self.total_collected_data += self.unit_cost
            elif self.cost.get(key, 0) < self.cost_sensor[s_key]: # Case when c(theta_s, theta_h) < b(theta_s), including c==0
                self.cost[key] = self.cost_sensor[s_key]
            else: # Case when c(theta_s, theta_h) == b(theta_s): collect one more unit
                self.cost[key] = min(self.cost[key] + self.unit_cost, self.max_cost)
                self.total_collected_data += self.cost[key] - self.cost_sensor[s_key]
                self.cost_sensor[s_key] = self.cost[key]

            # Total count of selected points
            self.count[key] = int(self.cost_sensor[s_key] / self.unit_cost)

            cost = self.cost[key]
            self.X_sample = np.vstack((self.X_sample, np.append(x, cost)))
            current_sample = x

            config = np.ravel_multi_index(np.asarray(x, dtype=int) - 1, dims) # Configuration index (0~8999)
            current_objective = objective_function(config, cost, sensor_scores) # ML performance of a configuration
            current_objective_max = objective_function(config, self.max_cost, sensor_scores) # ML performance at maximum budget
            self.Y_sample = np.append(self.Y_sample, -current_objective)
            self.total_evaluated_data += cost # Evaluated data
            self.total_collected_potential_data = self.total_collected_data + self.max_cost - cost # total maximum amount of collected data

            # Configuration index (1st row), ML performance at given budget (2nd row), ML performance at maximum budget (3rd row), total amount of collected data (4th row), total amount of evaluated data (5th row), total maximum amount of collected data (6th row)
            self.measurement = np.hstack((np.array(self.measurement).reshape(6,-1),
                                          np.array([config, current_objective, current_objective_max, self.total_collected_data, self.total_evaluated_data, self.total_collected_potential_data]).reshape(6,1)))

            print(f"[Iteration {iterations}] Score: {current_objective:.4f}, Data size: {cost}, Params: {current_sample}")

            if iterations >= self.n_initial_points-1:
                if self.surrogate == 'gp': # GP surrogate model
                    self.gp.fit(np.hstack((self.X_sample[:, :-1], self.X_sample[:, -1:] / self.max_cost)), self.Y_sample) # Normalize cost to [0, 1]
                elif self.surrogate == 'dpl': # DPL surrogate model
                    self.surrogate_model.fit(self.X_sample, self.Y_sample, max_cost=self.max_cost)
            
            iterations += 1

        best_idx = np.argmin(self.Y_sample)
        best_params = self.X_sample[best_idx][:-1] # Slicing to exclude a cost, which is the last column
        
        print("\n[Optimization Finished] Optimal Parameters are")
        
        return best_idx, best_params




# Search space of sensor parameters and hyperparameters
bounds = np.array([
    [0.5001, 3.4999],         # Step size [1, 2, 3]
    [0.5001, 5.4999],         # Start point [1, 2, 3, 4, 5]
    [0.5001, 4.4999],         # Number of points [1, 2, 3, 4, 5] instead of [1, 2, 3, 4]
    [0.5001, 5.4999],         # # of units in 1st hidden layer [2, 4, 6, 8, 10]
    [0.5001, 5.4999],         # # of units in 2nd hidden layer [2, 4, 6, 8, 10]
    [0.5001, 6.4999],         # Learning_rate [0.001, 0.0015] (1 is 0.001 ... 6 is 0.0015)
])

# Execute optimization
if __name__ == '__main__':
    
    optimizerMFBO = BayesianOptimizerMF(bounds=bounds, surrogate = 'gp', acq = 'ei', max_cost = 5000, unit_cost=1000, seed=0)
    best_idx, best_param = optimizerMFBO.optimize()
