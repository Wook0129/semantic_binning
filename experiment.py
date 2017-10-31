import numpy as np
import pandas as pd
import time

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as Agglo
from sklearn.metrics import normalized_mutual_info_score as NMI

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.preprocessing import LabelEncoder as LE

from data_handler import DataHandler
from semantic_binning import SemanticBinning


class Experiment:
    
    def __init__(self, data_path, var_dict, n_bins_range=range(2, 21),
                 batch_size=512, n_epoch=20, embedding_dim=16, lr=0.001, 
                 weight_decay=0.0, verbose=False, cv=10,
                 n_init_bins_list=[5, 10, 15, 20]):
        
        self.data = pd.read_csv(data_path)
        self.var_dict = var_dict
        
        self.n_bins_range = n_bins_range
        self.n_init_bins_list = n_init_bins_list

        self.semantic_binning = SemanticBinning(self.var_dict, batch_size=batch_size, n_epoch=n_epoch,
                                                embedding_dim=embedding_dim, lr=lr,
                                                weight_decay=weight_decay, verbose=verbose)
        self.cv = cv
        self.y = LE().fit_transform(self.data[var_dict['class_var']])

    def _get_classification_score(self, X):
        scores = []
        for clf in [LR(), NB(binarize=None), DT(), SVC(), RF()]:
            cv_scores = cross_val_score(clf, X, self.y, cv=self.cv, n_jobs=self.cv)
            mean, std = np.mean(cv_scores), np.std(cv_scores)
            scores.append((mean, std))
        return scores            
    
    def _get_clustering_score(self, X):
        n_cluster = len(set(self.y))
        scores = []
        for clstr in [KMeans(n_cluster)]:
            scores += [np.array([NMI(self.y, clstr.fit_predict(X)) for i in range(self.cv)]).mean()]
        return scores
    
    def perform_exp(self):
        
        start_time = time.time()
        
        list_of_scores = []
        
        data_handler = DataHandler(self.data, self.var_dict)
        
        raw_X = data_handler.get_dummy_coded_data('dummy_only')
        raw_clf_scores = self._get_classification_score(raw_X)
        raw_clstr_scores = self._get_clustering_score(raw_X)
        list_of_scores.append(('raw', raw_clf_scores, raw_clstr_scores, raw_X.shape[1]))
        
        n_cat_dummy_var = raw_X.shape[1] - len(self.var_dict['numerical_vars'])

        for n_bins in self.n_bins_range:
            ew_X = data_handler.get_dummy_coded_data('equal_width', n_bins)
            ew_clf_scores = self._get_classification_score(ew_X)
            ew_clstr_scores = self._get_clustering_score(ew_X)
            list_of_scores.append(('ew_{}'.format(n_bins), ew_clf_scores, ew_clstr_scores, ew_X.shape[1]-n_cat_dummy_var))
            
            ef_X = data_handler.get_dummy_coded_data('equal_freq', n_bins)
            ef_clf_scores = self._get_classification_score(ef_X)
            ef_clstr_scores = self._get_clustering_score(ef_X)
            list_of_scores.append(('ef_{}'.format(n_bins), ef_clf_scores, ef_clstr_scores, ef_X.shape[1]-n_cat_dummy_var))
        
        print(time.time() - start_time, 'training start')
        for n_init_bins in self.n_init_bins_list:
            sb_X = self.semantic_binning.fit_transform(self.data, n_init_bins)
            sb_clf_scores = self._get_classification_score(sb_X)
            sb_clstr_scores = self._get_clustering_score(sb_X)
            list_of_scores.append(('sb_{}'.format(n_init_bins), sb_clf_scores, sb_clstr_scores, sb_X.shape[1]-n_cat_dummy_var))

        return list_of_scores

    def print_scores(self, list_of_scores):
        
        exp_result = dict(disc_method=[], 
                          lr_acc=[], nb_acc=[], dt_acc=[],
                          svc_acc=[], rf_acc=[],
                          kmeans_nmi=[],
                          n_disc_cols=[])
        
        for x in list_of_scores:
            exp_result['disc_method'].append(x[0])
            exp_result['lr_acc'].append(x[1][0])
            exp_result['nb_acc'].append(x[1][1])
            exp_result['dt_acc'].append(x[1][2])
            exp_result['svc_acc'].append(x[1][3])
            exp_result['rf_acc'].append(x[1][4])
            exp_result['kmeans_nmi'].append(x[2][0])
            exp_result['n_disc_cols'].append(x[3])
            
        return pd.DataFrame(exp_result)
