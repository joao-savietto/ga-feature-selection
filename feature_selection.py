######################################################################
# IMPORTS

from genetic.algorithm import GeneticAlgorithm
from genetic.individual import Individual
from typing import List, Dict
import random 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

######################################################################
# READING DATASET & PREPROCESSING

dataset = pd.read_csv('Maths.csv')
dataset = dataset.dropna()
dataset = dataset.drop(['G2', 'G3'], axis=1)

categorical_features = []
numeric_features = []
for key in dict(dataset.dtypes):
    if dataset.dtypes[key] == 'object':
        categorical_features.append(key)
    else:
        numeric_features.append(key)

target = 'G1'
numeric_features.remove(target)        
dataset_categorical = dataset[categorical_features]
dataset_numeric = dataset[numeric_features]
dataset_pred = pd.concat([dataset_numeric, dataset_categorical], axis=1)
encoder = OneHotEncoder(categories='auto')
dummy_categorical = encoder.fit_transform(dataset_categorical).toarray()
dummy_categorical = pd.DataFrame(dummy_categorical, index=dataset_categorical.index)   
dataset_pred = pd.concat([dataset_numeric, dummy_categorical], axis=1)
scaler = MinMaxScaler()

######################################################################
# CROSS VALIDATION: BASE MODEL WITH ALL FEATURES
X = scaler.fit_transform(dataset_pred)
y = np.array(dataset[target])
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

model = LinearRegression()  
scores = cross_validate(model, X, y, cv=10, scoring=('neg_mean_squared_error'), return_train_score=True)
print('Base MSE: ', scores['test_score'].mean())

######################################################################
# GENETIC ALGORITHM INDIVIDUAL

class FeatureSelectionIndividual(Individual):
    def __init__(self, data: Dict[object, object], mutation_chance: float = 0.15):
        self.categorical_features = data['categorical_features']
        self.numeric_features = data['numeric_features']
        self.mutation_chance = mutation_chance
        super().__init__(data, mutation_chance)

    def random_chromossome(self) -> List[object]:
        self.all_features = self.categorical_features + self.numeric_features
        self.chromossome = [random.choice([0, 1]) for _ in range(len(self.all_features))]
        return self.chromossome

    def calculate_fitness(self):
        self.selected_categorical_features = [self.all_features[i] for i in range(len(self.chromossome)) if self.chromossome[i] == 1 and self.all_features[i] in self.categorical_features]
        self.selected_numeric_features = [self.all_features[i] for i in range(len(self.chromossome)) if self.chromossome[i] == 1 and self.all_features[i] in self.numeric_features]        
        if self.selected_categorical_features + self.selected_numeric_features == []:
            self.fitness = -1
            return -1
        
        self.dataset_pred = pd.DataFrame()
        self.dummy_categorical = pd.DataFrame()
        self.dataset_numeric = pd.DataFrame()

        if self.selected_categorical_features:
            self.dataset_categorical = dataset[self.selected_categorical_features]
            self.encoder = OneHotEncoder(categories='auto')
            self.dummy_categorical = self.encoder.fit_transform(self.dataset_categorical).toarray()
            self.dummy_categorical = pd.DataFrame(self.dummy_categorical, index=self.dataset_categorical.index)  
            self.dataset_pred = self.dummy_categorical
        
        if self.selected_numeric_features:
            self.dataset_numeric = dataset[self.selected_numeric_features]
         
        self.dataset_pred = pd.concat([self.dataset_numeric, self.dummy_categorical], axis=1)        
        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(self.dataset_pred)
        self.y = np.array(dataset[target])
        self.scaler_y = MinMaxScaler()
        self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1))
        self.model = LinearRegression()      
        self.scores = cross_validate(self.model, self.X, self.y, cv=10, scoring=('neg_mean_squared_error'))['test_score'].mean()
        self.fitness = sum([1 for x in self.chromossome if x == 0]) + self.scores
        if self.scores <= -0.0336:
            self.fitness = self.scores
        return self.fitness

    def mutate(self):
        reverse = lambda  x: 1 if x == 0 else 0 
        if np.random.random() < self.mutation_chance:
            self.chromossome = [reverse(self.chromossome[x]) if np.random.random() <= self.mutation_chance else self.chromossome[x] for x in range(len(self.all_features))]

######################################################################
# RUNNING GENETIC ALGORITHM

data = {'categorical_features': categorical_features, 'numeric_features': numeric_features}
base = FeatureSelectionIndividual(data)

algorithm = GeneticAlgorithm(base)
algorithm.run(population_size=200, early_stopping_tol=35)

######################################################################
# PRINTING RESULTS

best = algorithm.best_individual
features_kept = [best.all_features[i] for i in range(len(best.chromossome)) if best.chromossome[i] == 1]

print(f'{len(features_kept)} features kept out of {len(best.all_features)}')
print(f'Features kept: {features_kept}')

print('Base MSE: ', scores['test_score'].mean())
print(f'Final MSE: {best.scores}')