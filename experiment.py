import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score

from sklearn.preprocessing import StandardScaler

from data_handler import DataHandler
from semantic_binning import SemanticBinning


class Experiment:
    
    def __init__(self, data_path, var_dict, n_bins_range=range(2, 21),
                 batch_size=512, embedding_dim=16, max_iter=300000, verbose=False,
                 n_init_bins_list=[5, 10, 15, 20], random_state=42):
        
        self.data = pd.read_csv(data_path)
        self.var_dict = var_dict
        self.n_bins_range = n_bins_range
        self.n_init_bins_list = n_init_bins_list
        self.random_state = random_state
        
        self.semantic_binning = SemanticBinning(self.var_dict, batch_size=batch_size, embedding_dim=embedding_dim, max_iter=max_iter, verbose=verbose)
        self.class_var = self.data[var_dict['class_var']]
        self.n_class = len(self.class_var.unique())
    
    def _make_cv_folds(self, cv):

        cv_folds = []
        idxs = np.arange(0, len(self.data))
        
        np.random.seed(self.random_state)
        np.random.shuffle(idxs)

        folds = np.array_split(idxs, cv)

        for i in range(cv):
            train_idxs = np.concatenate([x for x in folds[:i] + folds[i+1:]])
            val_idxs = folds[i]
            cv_folds.append((train_idxs, val_idxs))

        return cv_folds

    def _get_classification_score(self, trn_x, val_x, trn_y, val_y, models):
            
        score = dict()

        if 'DT' in models:
            dt = DecisionTreeClassifier(min_samples_leaf=0.01, random_state=self.random_state)
            dt.fit(trn_x, trn_y)
            score['DT'] = accuracy_score(val_y, dt.predict(val_x))

        if 'LR' in models:
            lr = LogisticRegression(C=1.0, random_state=self.random_state)
            lr.fit(trn_x, trn_y)
            score['LR'] = accuracy_score(val_y, lr.predict(val_x))

        if 'NB' in models:
            nb = BernoulliNB(binarize=None)
            nb.fit(trn_x, trn_y)
            score['NB'] = accuracy_score(val_y, nb.predict(val_x))

        return score
        
    def classification_performance(self, cv=3):
        
        raw_scores, sc_scores = list(), list()
        ewb_scores, efb_scores, sb_scores = dict(), dict(), dict()
        
        for trn_idx, val_idx in self._make_cv_folds(cv):
            
            trn_x, val_x = self.data.loc[trn_idx], self.data.loc[val_idx]
            trn_y, val_y = self.class_var[trn_idx], self.class_var[val_idx]
            
            trn_data_handler = DataHandler(trn_x, self.var_dict)
            val_data_handler = DataHandler(val_x, self.var_dict)
            
            # Dummy Coding Only
            trn_raw = trn_data_handler.get_dummy_coded_data('dummy_only')
            val_raw = val_data_handler.get_dummy_coded_data('dummy_only')
            raw_score = self._get_classification_score(trn_raw, val_raw, trn_y, val_y, ['DT', 'LR'])
            raw_scores.append(raw_score)
            
            # Dummy Coding + Scaling for Numerical Vars
            scaler = StandardScaler()
            numerical_vars = self.var_dict['numerical_vars']
            trn_sc, val_sc = trn_x.copy(), val_x.copy()
            trn_sc[numerical_vars] = scaler.fit_transform(trn_sc[numerical_vars])
            val_sc[numerical_vars] = scaler.transform(val_sc[numerical_vars])
            trn_sc = DataHandler(trn_sc, self.var_dict).get_dummy_coded_data('dummy_only')
            val_sc = DataHandler(val_sc, self.var_dict).get_dummy_coded_data('dummy_only')
            
            sc_score = self._get_classification_score(trn_sc, val_sc, trn_y, val_y, ['DT', 'LR'])            
            sc_scores.append(sc_score)

            for n_bins in self.n_bins_range:
                
                # Equal Width Binning
                trn_ewb = trn_data_handler.get_dummy_coded_data('equal_width', n_bins)
                ewb_bins = trn_data_handler.get_bins_by_variable_from_data(trn_ewb)
                val_ewb = val_data_handler.get_dummy_coded_data(bins_by_variable=ewb_bins)
                ewb_score = self._get_classification_score(trn_ewb, val_ewb, trn_y, val_y, ['DT', 'LR', 'NB'])
                if n_bins not in ewb_scores:
                    ewb_scores[n_bins] = [ewb_score]
                else:
                    ewb_scores[n_bins].append(ewb_score)
                
                # Equal Freq Binning
                trn_efb = trn_data_handler.get_dummy_coded_data('equal_freq', n_bins)
                efb_bins = trn_data_handler.get_bins_by_variable_from_data(trn_efb)
                val_efb = val_data_handler.get_dummy_coded_data(bins_by_variable=efb_bins)
                efb_score = self._get_classification_score(trn_efb, val_efb, trn_y, val_y, ['DT', 'LR', 'NB'])
                if n_bins not in efb_scores:
                    efb_scores[n_bins] = [efb_score]
                else:
                    efb_scores[n_bins].append(efb_score)
                    
            for n_init_bins in self.n_init_bins_list:

                # Semantic Binning
                trn_sb = self.semantic_binning.fit_transform(trn_x, n_init_bins)
                val_sb = self.semantic_binning.transform(val_x)
                sb_score = self._get_classification_score(trn_sb, val_sb, trn_y, val_y, ['DT', 'LR', 'NB'])
                if n_init_bins not in sb_scores:
                    sb_scores[n_init_bins] = [sb_score]
                else:
                    sb_scores[n_init_bins].append(sb_score)
                    
        scores = dict(raw_scores=raw_scores, sc_scores=sc_scores,
                      ewb_scores=ewb_scores, efb_scores=efb_scores,
                      sb_scores=sb_scores)
        return scores
    
    def clustering_performance(self, methods=['kmeans', 'agglomerative']):
        
        def get_clustering_score(X, method):
            if method == 'kmeans':
                cluster_label = KMeans(n_clusters=self.n_class, 
                                       random_state=self.random_state).fit_predict(X)
            if method == 'agglomerative':
                cluster_label = AgglomerativeClustering(n_clusters=self.n_class).fit_predict(X)
            return normalized_mutual_info_score(self.class_var, cluster_label)
        
        scores = dict()
        
        data_handler = DataHandler(self.data, self.var_dict)
        
        scores['dummy_only'] = dict()
        dummy_coded = data_handler.get_dummy_coded_data('dummy_only')
        for method in methods:
            dummy_score = get_clustering_score(dummy_coded, method)
            scores['dummy_only'][method] = dummy_score
        
        scores['scale_numeric'] = dict()
        scale_numeric = data_handler.get_dummy_coded_data('scale_numeric')
        for method in methods:
            scale_score = get_clustering_score(scale_numeric, method)
            scores['scale_numeric'][method] = scale_score

        scores['equal_width'] = dict()
        for method in methods:
            scores['equal_width'][method] = dict()
            for n_bins in self.n_bins_range:
                ewb = data_handler.get_dummy_coded_data('equal_width', n_bins)
                ewb_score = get_clustering_score(ewb, method)
                scores['equal_width'][method][n_bins] = ewb_score

        scores['equal_freq'] = dict()
        for method in methods:
            scores['equal_freq'][method] = dict()
            for n_bins in self.n_bins_range:   
                efb = data_handler.get_dummy_coded_data('equal_freq', n_bins)
                efb_score = get_clustering_score(efb, method)
                scores['equal_freq'][method][n_bins] = efb_score
        
        scores['semantic_binning'] = dict()
        for method in methods:
            scores['semantic_binning'][method] = dict()
        for n_init_bins in self.n_init_bins_list:
            sb = self.semantic_binning.fit_transform(self.data, n_init_bins)
            for method in methods:
                sb_score = get_clustering_score(sb, method)
                scores['semantic_binning'][method][n_init_bins] = sb_score
                
        return scores

def compute_cv_score(scores, models=['DT', 'LR']):
    for model in models:
        print(model)
        cv_score = np.array([fold[model] for fold in scores])
        print('Accuracy = {0:0.3f} (+/- {1:0.3f})'.format(cv_score.mean(), cv_score.std() * 2))
        print('')

def compute_cv_score_by_n_bins(scores, models=['DT', 'LR', 'NB']):
    for model in models:
        print(model)
        for n_bins in scores:
            cv_score = np.array([fold[model] for fold in scores[n_bins]])
            print('#Bins = {0}, Accuracy = {1:0.3f} (+/- {2:0.3f})'.format(n_bins, cv_score.mean(), cv_score.std() * 2))
        print('')
        
def print_clustering_score(scores):
    for method, score in scores.items():
        print('{0}, NMI = {1:0.4f}'.format(method, score))
    
def print_clustering_score_by_n_bins(scores):
    for method, score_by_n_bins in scores.items():
        print(method)
        for n_bins, score in score_by_n_bins.items():
            print('#Bins = {0}, NMI = {1:0.4f}'.format(n_bins, score))