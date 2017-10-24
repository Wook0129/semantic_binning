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
                 batch_size=512, n_epoch=20, embedding_dim=16, lr=0.001, 
                 weight_decay=0.0, verbose=False,
                 n_init_bins_list=[5, 10, 15, 20], random_state=42):
        
        self.data = pd.read_csv(data_path)
        self.var_dict = var_dict
        self.n_cols = self.data.shape[1]
        if 'categorical_vars' in var_dict:
            self.n_dummy_coded_categorical_cols = pd.get_dummies(self.data[var_dict['categorical_vars']]).shape[1]
        else:
            self.n_dummy_coded_categorical_cols = 0
        self.n_bins_range = n_bins_range
        self.n_init_bins_list = n_init_bins_list
        self.random_state = random_state
        
        self.semantic_binning = SemanticBinning(self.var_dict, batch_size=batch_size, n_epoch=n_epoch, embedding_dim=embedding_dim, lr=lr, weight_decay=weight_decay, verbose=verbose)
        
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
        
    def test_classification_performance(self, cv=3):
        
        dummy_coded_scores, sc_scores = list(), list()
        ewb_scores, efb_scores, sb_scores = dict(), dict(), dict()
        
        for trn_idx, val_idx in self._make_cv_folds(cv):
            
            trn_x, val_x = self.data.loc[trn_idx], self.data.loc[val_idx]
            trn_y, val_y = self.class_var[trn_idx], self.class_var[val_idx]
            
            trn_data_handler = DataHandler(trn_x, self.var_dict)
            val_data_handler = DataHandler(val_x, self.var_dict)
            
            # Dummy Coding Only
            trn_dummy_coded = trn_data_handler.get_dummy_coded_data('dummy_only')
            val_dummy_coded = val_data_handler.get_dummy_coded_data('dummy_only')
            
            n_cols_dummy_coded = val_dummy_coded.shape[1]
            dummy_coded_score = self._get_classification_score(trn_dummy_coded, val_dummy_coded, trn_y, val_y, ['DT', 'LR'])
            dummy_coded_scores.append((n_cols_dummy_coded, dummy_coded_score))
            
            # Dummy Coding + Scaling for Numerical Vars
            scaler = StandardScaler()
            numerical_vars = self.var_dict['numerical_vars']
            trn_sc, val_sc = trn_x.copy(), val_x.copy()
            trn_sc[numerical_vars] = scaler.fit_transform(trn_sc[numerical_vars])
            val_sc[numerical_vars] = scaler.transform(val_sc[numerical_vars])
            trn_sc = DataHandler(trn_sc, self.var_dict).get_dummy_coded_data('dummy_only')
            val_sc = DataHandler(val_sc, self.var_dict).get_dummy_coded_data('dummy_only')
            
            n_cols_sc = val_sc.shape[1]
            sc_score = self._get_classification_score(trn_sc, val_sc, trn_y, val_y, ['DT', 'LR'])            
            sc_scores.append((n_cols_sc, sc_score))

            for n_bins in self.n_bins_range:
                
                # Equal Width Binning
                trn_ewb = trn_data_handler.get_dummy_coded_data('equal_width', n_bins)
                ewb_bins = trn_data_handler.get_bins_by_variable_from_data(trn_ewb)
                val_ewb = val_data_handler.get_dummy_coded_data(bins_by_variable=ewb_bins)
                
                n_cols_ewb = val_ewb.shape[1]
                ewb_score = self._get_classification_score(trn_ewb, val_ewb, trn_y, val_y, ['DT', 'LR', 'NB'])
                if n_bins not in ewb_scores:
                    ewb_scores[n_bins] = [(n_cols_ewb, ewb_score)]
                else:
                    ewb_scores[n_bins].append((n_cols_ewb, ewb_score))
                
                # Equal Freq Binning
                trn_efb = trn_data_handler.get_dummy_coded_data('equal_freq', n_bins)
                efb_bins = trn_data_handler.get_bins_by_variable_from_data(trn_efb)
                val_efb = val_data_handler.get_dummy_coded_data(bins_by_variable=efb_bins)
                
                n_cols_efb = val_efb.shape[1]
                efb_score = self._get_classification_score(trn_efb, val_efb, trn_y, val_y, ['DT', 'LR', 'NB'])
                if n_bins not in efb_scores:
                    efb_scores[n_bins] = [(n_cols_efb, efb_score)]
                else:
                    efb_scores[n_bins].append((n_cols_efb, efb_score))
                    
            for n_init_bins in self.n_init_bins_list:

                # Semantic Binning
                trn_sb = self.semantic_binning.fit_transform(trn_x, n_init_bins)
                val_sb = self.semantic_binning.transform(val_x)

                n_cols_sb = val_sb.shape[1]
                sb_score = self._get_classification_score(trn_sb, val_sb, trn_y, val_y, ['DT', 'LR', 'NB'])
                if n_init_bins not in sb_scores:
                    sb_scores[n_init_bins] = [(n_cols_sb, sb_score)]
                else:
                    sb_scores[n_init_bins].append((n_cols_sb, sb_score))
                    
        scores = dict(dummy_only=dummy_coded_scores, scale_numeric=sc_scores,
                      equal_width=ewb_scores, equal_freq=efb_scores,
                      semantic_binning=sb_scores)
        return scores
    
    def test_clustering_performance(self, methods=['kmeans', 'agglomerative']):
        
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
        n_cols_dummy_coded = dummy_coded.shape[1]
        for method in methods:
            dummy_score = get_clustering_score(dummy_coded, method)
            scores['dummy_only'][method] = (n_cols_dummy_coded, dummy_score)
        
        scores['scale_numeric'] = dict()
        scale_numeric = data_handler.get_dummy_coded_data('scale_numeric')
        n_cols_scale_numeric = scale_numeric.shape[1]
        for method in methods:
            scale_score = get_clustering_score(scale_numeric, method)
            scores['scale_numeric'][method] = (n_cols_scale_numeric, scale_score)

        scores['equal_width'] = dict()
        for method in methods:
            scores['equal_width'][method] = dict()
            for n_bins in self.n_bins_range:
                ewb = data_handler.get_dummy_coded_data('equal_width', n_bins)
                n_cols_ewb = ewb.shape[1]
                ewb_score = get_clustering_score(ewb, method)
                scores['equal_width'][method][n_bins] = (n_cols_ewb, ewb_score)

        scores['equal_freq'] = dict()
        for method in methods:
            scores['equal_freq'][method] = dict()
            for n_bins in self.n_bins_range:   
                efb = data_handler.get_dummy_coded_data('equal_freq', n_bins)
                n_cols_efb = efb.shape[1]
                efb_score = get_clustering_score(efb, method)
                scores['equal_freq'][method][n_bins] = (n_cols_efb, efb_score)
        
        scores['semantic_binning'] = dict()
        for method in methods:
            scores['semantic_binning'][method] = dict()
        for n_init_bins in self.n_init_bins_list:
            sb = self.semantic_binning.fit_transform(self.data, n_init_bins)
            n_cols_sb = sb.shape[1]
            for method in methods:
                sb_score = get_clustering_score(sb, method)
                scores['semantic_binning'][method][n_init_bins] = (n_cols_sb, sb_score)
                
        return scores

    def print_classification_scores(self, scores, method):

        if method in ['dummy_only', 'scale_numeric']:
            models = ['DT', 'LR']
            for model in models:
                print('{} performance'.format(model))
                n_cols = np.array([fold[0] for fold in scores[method]])
                cv_score = np.array([fold[1][model] for fold in scores[method]])
                print('#cols = {0}, Accuracy = {1:0.3f} (+/- {2:0.3f})'.format(n_cols.mean(), cv_score.mean(), cv_score.std() * 2))

        elif method in ['equal_width', 'equal_freq']:
            models = ['DT', 'LR', 'NB']
            for model in models:
                print('{} performance'.format(model))
                for n_bins in scores[method]:
                    n_cols = np.array([fold[0] for fold in scores[method][n_bins]])
                    cv_score = np.array([fold[1][model] for fold in scores[method][n_bins]])
                    print('#Bins = {0}, #Avg Cols = {1}, Accuracy = {2:0.3f} (+/- {3:0.3f})'.format(n_bins, n_cols.mean(), cv_score.mean(), cv_score.std() * 2))

        elif method == 'semantic_binning':
            models = ['DT', 'LR', 'NB']
            for model in models:
                print('{} performance'.format(model))
                for n_init_bins in scores[method]:
                    n_cols = np.array([fold[0] for fold in scores[method][n_init_bins]])
                    cv_score = np.array([fold[1][model] for fold in scores[method][n_init_bins]])
                    print('#Init Bins = {0}, #Avg Cols = {1}, Accuracy = {2:0.3f} (+/- {3:0.3f})'.format(n_init_bins, n_cols.mean(), cv_score.mean(), cv_score.std() * 2))
        else:
            raise NotImplementedError

    def print_clustering_scores(self, scores, method):
        
        if method in ['dummy_only', 'scale_numeric']:
            for algorithm, score in scores[method].items():
                print('{0}, #Cols = {1}, NMI = {2:0.4f}'.format(algorithm, score[0], score[1]))
        
        elif method in ['equal_width', 'equal_freq']:
            for algorithm, score_by_n_bins in scores[method].items():
                print(algorithm)
                for n_bins, score in score_by_n_bins.items():
                    print('#Bins = {0}, #Cols = {1}, #NMI = {2:0.4f}'.format(n_bins, score[0], score[1]))
        
        elif method == 'semantic_binning':
            for algorithm, score_by_n_bins in scores[method].items():
                print(algorithm)
                for n_init_bins, score in score_by_n_bins.items():
                    print('#Init Bins = {0}, #Cols = {1}, NMI = {2:0.4f}'.format(n_init_bins, score[0], score[1]))            
            
        else:
            raise NotImplementedError 
