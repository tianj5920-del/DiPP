"""CPU estimator worker. Transport is an anonymous parent/child pipe only.

Keep Torch out of this process: its OpenMP runtime can conflict
with binary builds of gradient-boosted tree libraries.
"""
import pickle
import sys
import warnings
import numpy as np


def fit(x, y, family, p, threads):
    if family.endswith('svm'):
        from sklearn.svm import SVC
        model=SVC(kernel='linear' if family=='linear_svm' else 'rbf',C=p['C'],gamma='scale',
                  class_weight='balanced',probability=True,random_state=101,cache_size=256)
    elif family in ('random_forest','extra_trees'):
        from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
        cls=RandomForestClassifier if family=='random_forest' else ExtraTreesClassifier
        model=cls(n_estimators=400,max_depth=p['depth'],min_samples_leaf=p['leaf'],max_features='sqrt',
                  class_weight='balanced',random_state=101,n_jobs=threads)
    elif family=='lightgbm':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=300,learning_rate=.05,max_depth=p['depth'],num_leaves=2**p['depth'],
            min_child_samples=10,reg_lambda=p['l2'],colsample_bytree=.8,subsample=.8,subsample_freq=1,
            random_state=101,n_jobs=threads,verbosity=-1)
    elif family=='xgboost':
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=300,learning_rate=.05,max_depth=p['depth'],min_child_weight=5,
            reg_lambda=p['l2'],colsample_bytree=.8,subsample=.8,objective='multi:softprob',eval_metric='mlogloss',
            tree_method='hist',device='cpu',random_state=101,n_jobs=threads)
    else:raise ValueError('Unknown classical estimator')
    options=dict(sample_weight=len(y)/(3*np.bincount(y,minlength=3)[y])) if family in ('lightgbm','xgboost') else {}
    model.fit(x,y,**options)
    if not np.array_equal(model.classes_,[0,1,2]):raise ValueError('Classifier column order mismatch')
    return model


def main():
    # Only invoked by this package's own Estimator; never load untrusted files.
    request=pickle.load(sys.stdin.buffer)
    if request['action']=='fit':
        model=fit(request['x'],request['y'],request['family'],request['parameters'],request['threads'])
        sys.stdout.buffer.write(pickle.dumps(model,protocol=5))
    elif request['action']=='predict':
        model=pickle.loads(request['model'])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='X does not have valid feature names')
            p=np.asarray(model.predict_proba(request['x']),dtype=float)
        pickle.dump(p,sys.stdout.buffer,protocol=5)
    else:raise ValueError('Unknown worker action')


if __name__=='__main__':main()
