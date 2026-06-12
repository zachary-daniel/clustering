import numpy as np
from sklearn.metrics import silhouette_score, confusion_matrix,classification_report
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import shap
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
'''
This file ocntains functions for doing data analysis between clusters. Will typically be used in the clustering notebooks that are one level up in the file structure.
i.e. TLCC_clustering.ipynb
'''


'''
This is a function for analyzing the distribution of data between n clusters

data_dict: dict of data of interest. typically the keys to this dict will event ids
cluster_dict: dictionary with each key corresponding to a different cluster, and each value being the event ids in that cluster
signals: if the dictionary is of the form (as it sometimes is) dict[key]['signals'] like the BES dict, then this flag cna be used
returns:
sorted_dict: dictionary with the keys as the clusters, and the values as the data being sorted through
'''
def cluster_comparison(data_dict, cluster_dict, signals = False):
    keys = [str(i) for i in cluster_dict.keys()] #keys from clusters
    sorted_dict = dict.fromkeys(keys, 0) #
    for i in sorted_dict.keys():
        sorted_dict[i] = []
    for id in data_dict.keys():
        if signals:
            data = data_dict[id]['signals']
        else:
            data = data_dict[id] #select data corresponding to current event id

        for cluster in cluster_dict.keys(): #cycle over all clusters
            if id in cluster_dict[cluster]: #if the current id is in the current cluster, add that data to the dictionary
                sorted_dict[cluster].append(data)
    return sorted_dict

'''
This function will do silhoutte score analysis on the clusters. Silhouette score is an average measure of how similar an element of one cluster is to the other members of the cluster,
and how different it is to the other clusters

dissimilarity_matrix: dissimilarity matrix for the elm database
max_clusters: max number of clusters to compute out to

returns:
sil_score_arr: array of silhouette scores for each number of clusters. first entry of the array corresponds to 2 clusters
'''

def silhouette_comparison(dissimilarity_matrix, max_clusters = 15):
     #these are just selected based off of the number of colors in the dendrogram for each linkage matrix

    linked_matrix = linkage(squareform(dissimilarity_matrix), method = 'ward') #need to transform from a distance matrix to a feature vector
    sil_score_arr = np.zeros(max_clusters-1)
    for num_clusters in range(max_clusters-1): #can't have fewer than 2 clusters
        labels_new = fcluster(linked_matrix, t = num_clusters + 2, criterion = 'maxclust')
        sil_score = silhouette_score(dissimilarity_matrix, labels_new, metric='precomputed')
        sil_score_arr[num_clusters] = sil_score
    return sil_score_arr


def to_dataframe(dict_list, names, ids):
    
    try:
        data = [dict_list[0][id]['signals'] for id in ids]
        df = pd.DataFrame(data, index = ids, columns = [names[0]])

    except:
        data = [dict_list[0][id] for id in ids]
        df = pd.DataFrame(data, index = ids, columns = [names[0]])

    for idx, dictionary in enumerate(dict_list):
        if idx == 0:
            continue
        
        if isinstance(dictionary[ids[0]], dict):
            next_key = list(dictionary[ids[0]].keys())[0]
            data = []
            for id in ids:
                if len(dictionary[id].keys()) != 0:
                    data.append(dictionary[id][next_key])
                else:
                    data.append(np.nan)
            df.insert(idx, names[idx], pd.Series(data, index = ids))
        else:
            data = []
            for id in ids:
                if isinstance(dictionary[id], str):
                    data.append(np.nan)
                else:
                    data.append(dictionary[id])
            df.insert(idx, names[idx], pd.Series(data, index = ids))
    return df

def tune_xgboost_random(X_train, y_train):
    """
    Imbalance-aware randomized search CV for XGBoost.
    """
    model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=42)
    
    param_dist = {
        "n_estimators": randint(50, 100),
        "max_depth": randint(2, 8),
        "learning_rate": uniform(0.01, 0.29), 
        "min_child_weight": randint(1, 8),
        "subsample": uniform(0.5, 0.5),      
        "colsample_bytree": uniform(0.5, 0.5)
    }
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20, 
        scoring='f1_macro', # Uses Macro F1 instead of accuracy
        cv=10,               
        verbose=3,
        n_jobs=1,          
        random_state=42
    )
    
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    search.fit(X_train, y_train, sample_weight=weights)
    return search.best_params_


def SHAP_analysis(features, targets, num_cat, test_size = .2, return_fig = False, run_optimize = True, **kwargs):
    '''
    This function is for determining which parameters and figures of merit (Torque, NBI, C coils, Beta etc.) are correlated to separations between clusters. The idea is as follows:
    a classifier is trained where the target is to predict the label of each piece of clustered data based on the external parameters that the hierarchical clustering did not have
    access to. Then these Shapley plots are generated which help determine which parameters have the largest effect on the separations

    features: these would be parameters like torque, beta, etc.
    targets: which id corresponds to which cluster
    test_size: fraction of data used for testing
    num_cat: number of clusters and categories
    **kwargs: parameters to pass to the xgboost classifier
    '''

    le = LabelEncoder()
    targets_encoded = le.fit_transform(targets)
    X_train, X_test, y_train, y_test = train_test_split(features, targets_encoded, test_size=test_size, random_state=42)
    
    xgb_params = {
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": True, # Reminder: Only True if your features have string/categorical columns!
        "random_state": 42,
        "max_depth": 3,               
        "min_child_weight": 1,        
        "gamma": 0.1,                 
        "subsample": 0.8,             
        "colsample_bytree": 0.8 
    }
    if run_optimize:
        best_params = tune_xgboost_random(X_train, y_train)
        # Update the base dictionary with the newly found best parameters
        xgb_params.update(best_params)

    #weights tuning
    weights_train = compute_sample_weight(class_weight='balanced', y=y_train)

    title = None
    if 'title' in kwargs:
        title = kwargs['title']
    model = xgb.XGBClassifier(**xgb_params)

    model.fit(X_train, y_train, sample_weight=weights_train)
    score = model.score(X_train, y_train)
    print(f'training score: {score}')
    cv_score = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'f1_macro')
    print(f'mean cv score: {np.mean(cv_score)}')
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cr = classification_report(y_test, y_pred)
    print(cr)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.initjs()
    print()
    axes = []
    figs = []
    if num_cat == 2:
        num_plot = 1
    else:
        num_plot = num_cat
    for k in range(num_plot):
        if num_cat == 2:
            shap.summary_plot(shap_values, X_test, show=False, max_display = 5)
        else:
            shap.summary_plot(shap_values[:, :, k], X_test, show=False, max_display = 5)
        for collection in plt.gca().get_children():
            if isinstance(collection, PathCollection):
                collection.set_sizes([50])
        if num_cat == 2:
            label = le.inverse_transform([1])[0] 
        else:
            label = le.inverse_transform([k])[0]
        if title == None: 
            plt.title(f"Cluster {label + 1}", fontsize=25, pad=20)
        else:
            plt.title(title, fontsize = 25, pad = 20)
        plt.gca().tick_params(axis = 'both', labelsize = 25)
        ax = plt.gca()
        ax.set_ylabel('')
        ax.set_xlabel('SHAP value (log likelihood)', fontsize = 25)
        fig = plt.gcf()
        fig.set_size_inches(8,8)
        cb_ax = fig.axes[-1]
        cb_ax.tick_params(labelsize = 18)
        cb_ax.set_ylabel('Feature Value', fontsize = 25)
        cb_ax.set_box_aspect(10) 
        cb_ax.set_anchor('E')

        plt.tight_layout()
        # plt.savefig(f'Paper_figures/beeswarm_plot_{k+1}_{pedestal_loc}.pdf',dpi=600)
        plt.show()
        
        
        figs.append(fig)
        axes.append(ax)
    if return_fig:
        return figs, axes

if __name__ == '__main__':

    #load in a bunch of parameters to use for the classifier



    elm_free_param_path = '../ELM_free_data_analysis/elm_free_param_dict.npy' 
    param_dict = np.load(elm_free_param_path, allow_pickle=True).item()
    q95_dict = {id: param_dict[id]['q95'] for id in param_dict.keys()}
    tritop_dict = {id: param_dict[id]['TRITOP'] for id in param_dict.keys()}
    tribot_dict = {id: param_dict[id]['TRIBOT'] for id in param_dict.keys()}
    kappa_dict = {id: param_dict[id]['KAPPA'] for id in param_dict.keys()}
    bt_ip = np.load('../elm_free_bes_data/elm_free_bt_dict.npy', allow_pickle=True).item()
    bt_dict = {id: bt_ip[id]['bt'] for id in bt_ip.keys()}
    ip_dict = {id: bt_ip[id]['ip'] for id in bt_ip.keys()}
    drsep_dict = np.load('../elm_free_bes_data/drsep_dict_elm_free.npy', allow_pickle=True).item()
    betan_dict = np.load('../elm_free_bes_data/beta_dict_elm_free.npy', allow_pickle = True).item()

    NBI_dict = np.load('../elm_free_bes_data/NBI_elm_free_dict.npy', allow_pickle=True).item()
    ECH_dict = np.load('../elm_free_bes_data/ECH_elm_free_dict.npy', allow_pickle=True).item()
    torque_dict = np.load('../elm_free_bes_data/NBI_torque_elm_free_dict.npy', allow_pickle=True).item()
    h98_dict = np.load('../elm_free_bes_data/H98_dict_elm_free.npy', allow_pickle = True).item()

    cluster_dict = np.load('../bes_foot_clusters_5.npy', allow_pickle=True).item()
    num_cat = len(cluster_dict.keys())
  
    

    dict_list = [q95_dict, tritop_dict, tribot_dict, kappa_dict, bt_dict, ip_dict, drsep_dict, betan_dict, NBI_dict, ECH_dict, torque_dict, h98_dict]
    names = ['q95', 'tritop', 'tribot', 'kappa', 'bt', 'ip', 'drsep', 'beta_n', 'NBI', 'ECH', 'torque', 'h98']

    cluster_dict_reshape = {}
    clusters = sorted(list(cluster_dict.keys()))
    cluster_conversion = {cluster: idx for idx,cluster in enumerate(clusters)}
    for cluster in clusters:
        ids = cluster_dict[cluster]
        temp = {id: cluster_conversion[cluster] for id in ids}
        cluster_dict_reshape.update(temp)
    cluster_df = pd.DataFrame({'Cluster': pd.Series(cluster_dict_reshape)})

    ids = list(cluster_dict_reshape.keys())
    nan_NBI = []
    nan_torque = []
    for id in ids:
        if len(NBI_dict[id].keys()) == 0:
            nan_NBI.append(id)
        if len(torque_dict[id].keys()) == 0:
            nan_torque.append(id)
        
    print(nan_NBI)
    

    feature_dataframe = to_dataframe(dict_list, names, ids)
    print(feature_dataframe)
    #need to reshape the cluster dict to have the ids as keys and the cluster as the value
    SHAP_analysis(feature_dataframe, cluster_df, num_cat)


