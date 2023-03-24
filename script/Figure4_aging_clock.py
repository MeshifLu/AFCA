import pandas as pd
from anndata import AnnData
import numpy as np
import scanpy as sc
from collections import Counter
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

adata = sc.read("afcaFca_headBody_stringent.h5ad")

def check_num_cells(age, thr_total=500, thr_stage=200):
    if len(age) < thr_total:
        return False
    cc = Counter(age)
    for k in cc.keys():
        if cc[k]<thr_stage:
            return False     
    return True

def get_aging_genes(adata, importance, corr_thr=0.3):
    iscores = {}
    aging_genes = []
    for i,g in enumerate(importance):
        if np.abs(g)>0:
            iscores[adata.var_names[i]] = g
    
    for gene in iscores.keys():
        exp_data = np.array(adata[:,gene].X.todense()).squeeze()
        
        age_data = [float(y1) for y1 in adata.obs['age']]
        corr, _ = pearsonr(exp_data, age_data)
        
        if np.abs(corr)>corr_thr:
            aging_genes.append(gene)

    return aging_genes

def regress_on_age(adata, find_aging_genes=False):

    cell_types = sorted(list(set(adata.obs['afca_annotation'])))
    all_R2 = []
    ct_aging_genes = {}
    pearson_thr=0.3
    
    for cell_type in cell_types:
               
        if cell_type == 'unannotated':
            continue
            
        adata_ct = adata[adata.obs['afca_annotation']==cell_type]
        y = adata_ct.obs['age'].values
        
        if not check_num_cells(y):
            continue
        
        y = [float(y1) for y1 in y]
        X = adata_ct.X.todense()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        elnet = ElasticNet(random_state=0, l1_ratio=0.1, max_iter=1000)
        elnet.fit(X_train, y_train)
        preds = elnet.predict(X_test)
        importance = elnet.coef_
        
        R2 = elnet.score(X_test, y_test, sample_weight=None)
        print('{} {} {}'.format(cell_type, len(y), R2))
        all_R2.append(R2)
        
        if find_aging_genes:
            aging_genes = get_aging_genes(adata_ct, importance, pearson_thr)
            ct_aging_genes[cell_type] = aging_genes

    print('Mean {}'.format(np.mean(all_R2)))
    
    return all_R2, ct_aging_genes

regress_on_age(adata)