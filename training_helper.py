import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.utils import resample

from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

MODEL_LOGIT = 'logit'
MODEL_GRADIENT = 'gradient'
MODEL_FOREST = 'forest'

FEAT_SELECT_ANOVA = 'anova'
FEAT_SELECT_CHI2 = 'chi2'

# Compute the bootstrap confidence interval of given data at
# a specific percentage (between 0 and 100) applied on some stat function
def bootstrap_ci(data, nb_samples, nb_iteration, percentage, stat_function):
    stats = []
    for i in range(nb_iteration):
        # Take randomly n_samples samples with replacement
        samples = resample(data, replace=True, n_samples=nb_samples)
        stats.append(stat_function(samples)) # Compute the stat of the distribution
    
    # Compute the confidence interval for the given percentage
    lower = np.percentile(stats, (100 - percentage)/2)
    upper = np.percentile(stats, percentage + ((100 - percentage)/2))
    
    return lower, upper

# Cross-validation step specific to the logistic regression model
def logit_cross_validation(X, y, n_folds):
    kf = KFold(n_folds)
    accs = []
    for train_index, test_index in kf.split(X):
        # Train the model
        model = Logit(y[train_index], add_constant(X[train_index])).fit(maxiter=50000, disp=0) # A high max-iteration is required to make sure we converged in all situation
        y_pred = model.predict(add_constant(X[test_index]))
        acc = accuracy_score(y[test_index], np.round(y_pred))
        accs.append(acc)
    
    # Compute the mean accuracy and the 95% confidence interval
    mean_acc = np.mean(accs)    
    low_acc, high_acc = bootstrap_ci(accs, len(accs), 1000, 95, np.mean)

    return mean_acc, low_acc, high_acc

# Cross-validation step specific to the gradient boosting regressor model
def gradient_cross_validation(X, y, n_folds, n_estimators, learning_rates):
    best_acc = 0
    for n_esti in n_estimators: # Try many number of estimators
        for lr in learning_rates: # Try many learning rate
            kf = KFold(n_folds)
            accs = []
            for train_index, test_index in kf.split(X):
                # Train the model
                model = GradientBoostingRegressor(n_estimators = n_esti, learning_rate = lr).fit(X[train_index], y[train_index])
                y_pred = model.predict(X[test_index])
                acc = accuracy_score(y[test_index], np.round(y_pred))
                accs.append(acc)

            # Compute the mean accuracy and the 95% confidence interval
            mean_acc = np.mean(accs)    
            low_acc, high_acc = bootstrap_ci(accs, len(accs), 1000, 95, np.mean)

            # Select the best number of estimators and learning rate based on the mean accuracy
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_low = low_acc
                best_high = high_acc

                best_n_esti = n_esti
                best_lr = lr

    return best_acc, best_low, best_high, [best_n_esti, best_lr]

# Cross-validation step specific to the random forests model
def forest_cross_validation(X, y, n_folds, n_estimators):
    best_acc = 0
    for n_esti in n_estimators: # Try many number of estimators
        kf = KFold(n_folds)
        accs = []
        for train_index, test_index in kf.split(X):
            # Train the model
            model = RandomForestRegressor(n_estimators = n_esti).fit(X[train_index], y[train_index])
            y_pred = model.predict(X[test_index])
            acc = accuracy_score(y[test_index], np.round(y_pred))
            accs.append(acc)

        # Compute the mean accuracy and the 95% confidence interval
        mean_acc = np.mean(accs)    
        low_acc, high_acc = bootstrap_ci(accs, len(accs), 1000, 95, np.mean)

        # Select the best number of estimators based on the mean accuracy
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_low = low_acc
            best_high = high_acc

            best_n_esti = n_esti

    return best_acc, best_low, best_high, [best_n_esti]

# Train a given model with cross-validation to select the best model parameters and return the accuracy of that model
def train_and_evaluate(model_type, feat_select_type, X, y, n_folds, nb_features, n_estimators=None, learning_rates=None):
    
    # Select the algorithm corresponding the feature selection type
    if feat_select_type == FEAT_SELECT_ANOVA:
        feat_select_alg = f_classif
    elif feat_select_type == FEAT_SELECT_CHI2:
        feat_select_alg = chi2
    else:
        feat_select_alg = None

    # Perform the cross-validation
    best_acc = 0
    stop_after_iter = False # No need to keep iterating on the number of features to select if true
    for nb_feat in tqdm(nb_features):
        if stop_after_iter: 
            continue # Do nothing to complete the progress bar execution
            
        # If the number of features is too high, select all features
        if nb_feat > len(X.columns):
            nb_feat = 'all'
            stop_after_iter = True # Next iteration will have high number of feature so no need to keep iterating
            
        if feat_select_alg != None:
            # Select the best nb_feat features with the selected algorithm
            selector = SelectKBest(feat_select_alg, k=nb_feat)
            selector.fit(X, y)

            new_X = selector.transform(X)
        else:
            nb_feat = 'all' # No features selection so take all features
            new_X = X.to_numpy()
            stop_after_iter = True # No need to iterate multiple times

        # Perform the specific step of cross validation for each models and get their accuracy
        if model_type == MODEL_LOGIT:
            acc, low_acc, high_acc = logit_cross_validation(new_X, y.to_numpy(), n_folds)
            model_params = []
        elif model_type == MODEL_GRADIENT:
            acc, low_acc, high_acc, model_params = gradient_cross_validation(new_X, y.to_numpy(), n_folds, n_estimators, learning_rates)
        elif model_type == MODEL_FOREST:
            acc, low_acc, high_acc, model_params = forest_cross_validation(new_X, y.to_numpy(), n_folds, n_estimators)

        # Select the best model parameters based on the accuracy
        if acc > best_acc:
            best_acc = acc
            best_low = low_acc
            best_high = high_acc
            best_model_params = [nb_feat] + model_params
        
    return best_acc, best_low, best_high, best_model_params