import numpy as np
import pandas as pd
import time
import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.preprocessing import LabelEncoder as LE

from data_handler import DataHandler
from semantic_binning import SemanticBinning


class Experiment:
    
    def __init__(self, data_path, var_dict, n_bins_range=range(2, 21),
                 batch_size=512, n_epoch=20, embedding_dim=16, lr=0.001, 
                 weight_decay=0.0, verbose=False, cv=10,
                 n_init_bins_list=[5, 10, 15, 20],
                 co_occur_cutoff=1,):
        
        self.data = pd.read_csv(data_path)
        self.var_dict = var_dict
        
        self.n_bins_range = n_bins_range
        self.n_init_bins_list = n_init_bins_list

        self.semantic_binning = SemanticBinning(self.var_dict, batch_size=batch_size, n_epoch=n_epoch,
                                                embedding_dim=embedding_dim, lr=lr,
                                                weight_decay=weight_decay, verbose=verbose,
                                                co_occur_cutoff=co_occur_cutoff)
        self.cv = cv
        self.y = LE().fit_transform(self.data[var_dict['class_var']])
        
        self.lr_params = [0.5,1.0]
        self.dt_params = [3,4,5]
        self.rf_params = [10,20,30]
        
    def _get_classification_score(self, X):
        
        def get_cv_score(clf, X):
            cv_scores = cross_val_score(clf, X, self.y, cv=self.cv, n_jobs=self.cv)
            mean, std = np.mean(cv_scores), np.std(cv_scores)
            return round(mean, 3), round(std, 3)
        
        scores = dict()

        for c in self.lr_params:
            scores['lr_acc_C={}'.format(c)] = get_cv_score(LR(C=c), X)
        for d in self.dt_params:
            scores['dt_acc_depth={}'.format(d)] = get_cv_score(DT(max_depth=d), X)
        for n in self.rf_params:
            scores['rf_acc_n_est={}'.format(n)] = get_cv_score(RF(n_estimators=n), X)
        
        return scores
    
    def perform_exp(self):
        
        list_of_scores = []
        
        data_handler = DataHandler(self.data, self.var_dict)
        
        raw_X = data_handler.get_dummy_coded_data('dummy_only')
        n_cat_dummy_var = raw_X.shape[1] - len(self.var_dict['numerical_vars'])
        
        raw_clf_scores = self._get_classification_score(raw_X)
        list_of_scores.append(('raw', raw_clf_scores, raw_X.shape[1] - n_cat_dummy_var))

        for n_init_bins in self.n_init_bins_list:
            sb_X = self.semantic_binning.fit_transform(self.data, n_init_bins)
            sb_clf_scores = self._get_classification_score(sb_X)
            list_of_scores.append(('sb_{}'.format(n_init_bins), sb_clf_scores, sb_X.shape[1]-n_cat_dummy_var))
        
        for n_bins in self.n_bins_range:
            ew_X = data_handler.get_dummy_coded_data('equal_width', n_bins)
            ew_clf_scores = self._get_classification_score(ew_X)
            list_of_scores.append(('ew_{}'.format(n_bins), ew_clf_scores, ew_X.shape[1]-n_cat_dummy_var))
            
            ef_X = data_handler.get_dummy_coded_data('equal_freq', n_bins)
            ef_clf_scores = self._get_classification_score(ef_X)
            list_of_scores.append(('ef_{}'.format(n_bins), ef_clf_scores, ef_X.shape[1]-n_cat_dummy_var))
        
        self.list_of_scores = list_of_scores
        print('Experiment Finished !. Result Saved in Exp Instance..')

    def get_result(self):
        
        exp_result = dict()
        
        exp_result['disc_method'] = []
        
        for C in self.lr_params:
            exp_result['lr_acc_C={}'.format(C)] = []
        for depth in self.dt_params:
            exp_result['dt_acc_depth={}'.format(depth)] = []
        for n_est in self.rf_params:
            exp_result['rf_acc_n_est={}'.format(n_est)] = []

        exp_result['n_disc_cols'] = []
        
        for x in self.list_of_scores:
            
            exp_result['disc_method'].append(x[0])
            
            for C in self.lr_params:
                clf = 'lr_acc_C={}'.format(C)
                exp_result[clf].append(x[1][clf][0])
                
            for d in self.dt_params:
                clf = 'dt_acc_depth={}'.format(d)
                exp_result[clf].append(x[1][clf][0])

            for n in self.rf_params:
                clf = 'rf_acc_n_est={}'.format(n)
                exp_result[clf].append(x[1][clf][0])
                
            exp_result['n_disc_cols'].append(x[2])
            
        return pd.DataFrame(exp_result)

    def report_best_result_by_method(self):
        best_result = self.get_result()
        for model in ['dt','lr','rf']:
            acc_by_parameter = [col for col in best_result.columns if model+'_acc' in col]
            best_result[model] = best_result[acc_by_parameter].max(axis=1)
            best_result = best_result.drop(acc_by_parameter, axis=1)
        return best_result

    def plot_model_comparison_chart(self, result):

        def plot_chart_for_model(loc_x, loc_y, title, acc, ylabel):

            ax[loc_x][loc_y].set_title(title, fontsize=25)

            raw_ncols, raw_acc = [], []
            ew_ncols, ew_acc = [], []
            ef_ncols, ef_acc = [], []
            sb_ncols, sb_acc = [], []
            
            for c, a, d in zip(rel_n_cols, acc, disc_method):
                if 'raw' in d:
                    raw_ncols.append(c)
                    raw_acc.append(a)
                if 'ew' in d:
                    ew_ncols.append(c)
                    ew_acc.append(a)
                if 'ef' in d:
                    ef_ncols.append(c)
                    ef_acc.append(a)
                if 'sb' in d:
                    sb_ncols.append(c)
                    sb_acc.append(a)
                    
            ew = ax[loc_x][loc_y].scatter(x=ew_ncols, y=ew_acc, s=200, c='g', marker='x')
            ef = ax[loc_x][loc_y].scatter(x=ef_ncols, y=ef_acc, s=200, c='b', marker='x')
            sb = ax[loc_x][loc_y].scatter(x=sb_ncols, y=sb_acc, s=200, c='r', marker='x')
            
            if title != 'Naive Bayes':
                raw = ax[loc_x][loc_y].scatter(x=raw_ncols, y=raw_acc, s=200, c='y', marker='x')
                points = (raw, ew, ef, sb)
                legend_names = ('Raw', 'Equal Width', 'Equal Freq', 'Proposed')
            
            else:
                points = (ew, ef, sb)
                legend_names = ('Equal Width', 'Equal Freq', 'Proposed')                
            ax[loc_x][loc_y].legend(points, legend_names, 
                                    scatterpoints=1, loc='lower right', ncol=2, fontsize=12)        
            for i, xy in enumerate(zip(rel_n_cols, acc)):
                ax[loc_x][loc_y].annotate(disc_method[i], xy, fontsize=15)

            ax[loc_x][loc_y].set_xlabel('Relative Number of Columns', fontsize=20)
            ax[loc_x][loc_y].set_ylabel(ylabel, fontsize=20)

        disc_method = result['disc_method']

        n_cols = result['n_disc_cols']
        max_n_cols = n_cols.max()
        rel_n_cols = [(x / max_n_cols) for x in n_cols]

        fig, ax = plt.subplots(figsize=(20, 30), ncols=1, nrows=3)
        plt.suptitle('Classification Performances', fontsize=30)

        plot_chart_for_model(0, 0, 'Decision Tree(depth=3)', result['dt_acc_depth=3'], 'Accuracy')
        plot_chart_for_model(0, 1, 'Logistic Regression(C=1.0)', result['lr_acc_C=1.0'], 'Accuracy')
        plot_chart_for_model(0, 2, 'Naive Bayes', result['nb_acc'], 'Accuracy')

    def plot_pairwise_distance_matrices(self):

        fig_ncols = 3
        fig_nrows = int(np.ceil(len(self.var_dict['numerical_vars']) / fig_ncols))

        fig, ax = plt.subplots(figsize=(fig_ncols * 10, fig_nrows * 10), ncols=fig_ncols, nrows=fig_nrows)
        plt.suptitle('Pairwise-distance between embeddings', fontsize=30)

        for i, var in enumerate(self.var_dict['numerical_vars']):

            row = i // fig_ncols
            col = i - fig_ncols * row

            ax[row][col].set_title(var, fontsize=25)
            cols, dist_matrix = self.semantic_binning._bin_merger._get_cols_and_pairwise_dist_btw_embeddings(var)

            cols = ['(' + x.split('_(')[1] for x in cols]
            split_points = self.semantic_binning.bins_by_var[var]['split_point']
            begin_end_points = [[float(x) for x in col.replace('(','').replace(']','').split(', ')] for col in cols]
            cluster_label = []
            c_label = 1
            for j, (begin, end) in enumerate(begin_end_points):
                if end <= split_points[c_label]:
                    cluster_label.append(c_label)
                else:
                    c_label += 1
                    cluster_label.append(c_label)
                    ax[row][col].axhline(j, c="r", linewidth=5)
                    ax[row][col].axvline(j, c="r", linewidth=5)

            sns.heatmap(dist_matrix, cmap='gray', xticklabels=cluster_label,
                        yticklabels=cols, ax=ax[row][col])
        