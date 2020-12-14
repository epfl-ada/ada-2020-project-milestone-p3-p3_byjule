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

def logit_cross_validation(X, y, n_folds):
    kf = KFold(n_folds)
    accs = []
    for train_index, test_index in kf.split(X):
        model = Logit(y[train_index], add_constant(X[train_index])).fit(maxiter=50000, disp=0)
        y_pred = model.predict(add_constant(X[test_index]))
        acc = accuracy_score(y[test_index], np.round(y_pred))
        accs.append(acc)
        
    mean_acc = np.mean(accs)    
    low_acc, high_acc = bootstrap_ci(accs, len(accs), 1000, 95, np.mean)

    return mean_acc, low_acc, high_acc

def gradient_cross_validation(X, y, n_folds, n_estimators, learning_rates):
    best_acc = 0
    for n_esti in n_estimators:
        for lr in learning_rates:
            kf = KFold(n_folds)
            accs = []
            for train_index, test_index in kf.split(X):
                model = GradientBoostingRegressor(n_estimators = n_esti, learning_rate = lr).fit(X[train_index], y[train_index])
                y_pred = model.predict(X[test_index])
                acc = accuracy_score(y[test_index], np.round(y_pred))
                accs.append(acc)

            mean_acc = np.mean(accs)    
            low_acc, high_acc = bootstrap_ci(accs, len(accs), 1000, 95, np.mean)

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_low = low_acc
                best_high = high_acc

                best_n_esti = n_esti
                best_lr = lr

    return best_acc, best_low, best_high, [best_n_esti, best_lr]

def forest_cross_validation(X, y, n_folds, n_estimators):
    best_acc = 0
    for n_esti in n_estimators:
        kf = KFold(n_folds)
        accs = []
        for train_index, test_index in kf.split(X):
            model = RandomForestRegressor(n_estimators = n_esti).fit(X[train_index], y[train_index])
            y_pred = model.predict(X[test_index])
            acc = accuracy_score(y[test_index], np.round(y_pred))
            accs.append(acc)

        mean_acc = np.mean(accs)    
        low_acc, high_acc = bootstrap_ci(accs, len(accs), 1000, 95, np.mean)

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_low = low_acc
            best_high = high_acc

            best_n_esti = n_esti

    return best_acc, best_low, best_high, [best_n_esti]

def train_and_evaluate(model_type, feat_select_type, X, y, n_folds, nb_features, n_estimators=None, learning_rates=None):
    
    if feat_select_type == FEAT_SELECT_ANOVA:
        feat_select_alg = f_classif
    elif feat_select_type == FEAT_SELECT_CHI2:
        feat_select_alg = chi2
    else:
        feat_select_alg = None

    best_acc = 0
    stop_after_iter = False
    for nb_feat in tqdm(nb_features):
        if stop_after_iter:
            continue
            
        if nb_feat > len(X.columns):
            nb_feat = 'all'
            stop_after_iter = True
            
        if feat_select_alg != None:
            selector = SelectKBest(feat_select_alg, k=nb_feat)
            selector.fit(X, y)

            new_X = selector.transform(X)
        else:
            nb_feat = 'all'
            new_X = X.to_numpy()
            stop_after_iter = True

        if model_type == MODEL_LOGIT:
            acc, low_acc, high_acc = logit_cross_validation(new_X, y.to_numpy(), n_folds)
            model_params = []
        elif model_type == MODEL_GRADIENT:
            acc, low_acc, high_acc, model_params = gradient_cross_validation(new_X, y.to_numpy(), n_folds, n_estimators, learning_rates)
        elif model_type == MODEL_FOREST:
            acc, low_acc, high_acc, model_params = forest_cross_validation(new_X, y.to_numpy(), n_folds, n_estimators)

            
        if acc > best_acc:
            best_acc = acc
            best_low = low_acc
            best_high = high_acc
            best_model_params = [nb_feat] + model_params
        
    return best_acc, best_low, best_high, best_model_params