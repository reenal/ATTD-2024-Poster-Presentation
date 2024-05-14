"""
##################################################
## Script for estimating individual risk-scores. Required for Table-1
##################################################
# Author: Shahrukh Iqbal
# Email: iqbal@mainly.ai
##################################################
"""


import sys
sys.path.append("../")
from pathlib import Path
import pickle
from config import Config

# numeric
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd
from functools import reduce

# import warnings
# from scipy.integrate import IntegrationWarning
# warnings.filterwarnings(action="ignore", category= IntegrationWarning)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 70)
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.integrate import quad
from sklearn import set_config
set_config(display="text")  # displays text representation of estimators

from tqdm import tqdm




class EstimateAntibodySignalProbability:
    """
    Estimates the Probabilty for a given MaskID/DRAW_AGE/Antibody based on past observation. This is the Probability P(x_t|x_{t-1}) for
    any given autoantibody.
    """
    def __init__(self) -> None:
        pass
    def markov_model(self, stream, model_order):
        """
        Stream is a list of binary values and model_order is n, 
        when you want to calculate conditional probability of having x_{t} as next signal when you are given x_1, x_2, ..., x_{t-1}.
        This function returns model and stats. 
        stats is a dictionary counter giving counts of occurences of each n group.
        model is a dictionary which gives number of occurences of nth signal for given t-1 signal group in variable stats."""
        model, stats = defaultdict(Counter), Counter()
        circular_buffer = deque(maxlen = model_order)
        for token in stream:
            prefix = tuple(circular_buffer)
            circular_buffer.append(token)
            if len(prefix) == model_order:
                stats[prefix] += 1
                model[prefix][token] += 1
        return model, stats
    
    def initial_prob(self, arr):
        """
        For handling edge cases. Especially those encountered for smaller series and with results for 1st two timestamps.
        """
        p_zero = np.mean(arr==0)
        p_one= 1- p_zero

        dict_prob = {0: p_zero, 1: p_one}
        intial_prob = dict_prob[arr[0]]
        return intial_prob
    def get_last_visit_prob(self,stream):
        """
        Assuming a \tau=1 for the markov chain.
        So P(x_t| (x_1,..., x_{t-1})) = P(x_t|x_{t-1})
        """
        if len(stream) <2:
            return 0
        elif len(stream)==2:
            in_prob = self.initial_prob(stream)
            return (1-in_prob)*in_prob
        else:
            _ , stats = self.markov_model(stream=stream, model_order=2)
            (x,y) = stream[-2:]
            prob = stats[(x,y)]/sum(stats.values())
            return prob

class EstimateTimeWindowProbabilityDensity:
    """
    Creates a dictionary for storing Pr(t<=T) between the 'DRAW_AGE' 365 days and 5160 days.
    """
    def __init__(self, model) -> None:
        self.model = model
        self.prob_dict = dict()
        self.one_day_time_window = list(map(lambda x: (x,x+1), np.arange(365, 5160))) # Window is between 1-year (365) and maximum observed DRAW_AGE (5160)
    def estimate_prob(self, x):
        """Utilized the KernelDensity() model for estimating the Probability density at time x.
        Params:
            x[float]: DRAW_AGE in days
        """
        return np.exp(self.model.score_samples(np.array(x).reshape(-1,1)))[0]
    def estimate_prob_density(self, values):
        """
        Estimates Pr(low<=t<=high) by integrating Probability density between 'low' and 'high'
        """
        low, high = values
        low = float(low) if low>=365 else float(365) # Our interim dataset consists of DRAW_AGE>=365 days
        high = float(high)
        key = (low,high)
        if key not in self.prob_dict.keys():
            if not (low == high):
                prob_density = quad(func=self.estimate_prob, a=low, b=high, limit=1000)[0] # Setting limit=1000 reduces the estimation error
            else:
                prob_density = 0.
            self.prob_dict[(low,high)] = np.round(prob_density, decimals=6) # Update the data-dictionary
            return self.prob_dict[(low,high)]
        elif (key in self.prob_dict.keys()):
            return self.prob_dict[(low,high)]


class DrawAgeProbability:
    def __init__(self, prob_dict) -> None:
        
        self.prob_dict = prob_dict
    def multi_day_time_window(self, start_age, end_age):
        """
        Returns list of estimated Probability between ´start_age´ and ´end_age´ as [Pr(start_age<=t<=start_age+1),..., Pr(end_age-1<=t<=end_age)]
        """
        windows = map(lambda x: (x,x+1), np.arange(start_age, end_age))
        return windows
    def time_window_density(self, draw_ages):
        """
        Function estimates that for a given MaskID, what is the probability of occuring between the window of ´draw_ages´
        """
        assert draw_ages[0]<= draw_ages[1], "Ensure order of DRAW_AGE is correct"
        prob = np.sum([self.prob_dict[key] for key in self.multi_day_time_window(draw_ages[0], draw_ages[1])]).round(6)
        return prob
