import numpy as np
from sklearn import metrics
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import train_test_split

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.1
param['max_depth'] = 3
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.8
param['colsample_bytree'] = 0.2
param['seed'] = 0
param['gamma'] = 0.2
num_rounds=500

param_ = {}
param_['objective'] = 'multi:softprob'
param_['eta'] = 0.1
param_['max_depth'] = 3
param_['silent'] = 1
param_['num_class'] = 3
param_['eval_metric'] = "mlogloss"
param_['min_child_weight'] = 1
param_['subsample'] = 0.8
param_['colsample_bytree'] = 0.7
param_['seed'] = 0
param_['gamma'] = 0.2
num_rounds_=69

param_lgmb = {}
param_lgmb['objective']='multiclass'
param_lgmb['num_class']=3
param_lgmb['metric']='multi_logloss'
param_lgmb['learning_rate']=0.1
param_lgmb['num_leaves']=15
param_lgmb['max_depth']=5
param_lgmb['seed']=2017

class ensemble():
    """author identification最终集成"""
    def __init__(self,estimators):
        self.estimators = []
        self.estimators_name = []
        for i in estimators:
            self.estimators_name.append(i[0])
            self.estimators.append(i[1])
        self.clf = xgb

    def fit(self,x_train,y_train):
        x = []
        for i in self.estimators:
            print(self.estimators_name[self.estimators.index(i)])
            if self.estimators_name[self.estimators.index(i)]!='xgb' and self.estimators_name[self.estimators.index(i)]!='lgbm':
                i.fit(x_train,y_train)
                result = i.predict_proba(x_train)
                for j in range(3):
                    x.append(result[:,j])
            else:
                if self.estimators_name[self.estimators.index(i)] == 'lgbm':
                    lgbm_x_val,lgbm_x_dva,lgbm_y_val,lgbm_y_dva = train_test_split(x_train,y_train,test_size=0.33,random_state=2017)
                    lgbm_train = lgbm.Dataset(lgbm_x_val,lgbm_y_val)
                    lgbm_test = lgbm.Dataset(lgbm_x_dva,lgbm_y_dva,reference=lgbm_train)
                    self.lgbm_model = i.train(params=param_lgmb,train_set=lgbm_train,valid_sets=lgbm_test,num_boost_round=num_rounds,early_stopping_rounds=20)
                    result = self.lgbm_model.predict(x_train,num_iteration=self.lgbm_model.best_iteration)
                    for j in range(3):
                        x.append(result[:,j])
                    print(result)
                    print(len(result))
                else:
                    self.xgb_model = i.train(params=list(param.items()),dtrain=xgb.DMatrix(pd.DataFrame(x_train),pd.DataFrame(y_train)),num_boost_round=num_rounds)
                    result = self.xgb_model.predict(xgb.DMatrix(pd.DataFrame(x_train)))
                    for j in range(3):
                        x.append(result[:,j])
        x = np.array(x).T
        y = y_train
        self.model = self.clf.train(num_boost_round=num_rounds_,params=list(param_.items()),dtrain=xgb.DMatrix(pd.DataFrame(x),pd.DataFrame(y)))
        print(self.model)
        print("训练完成")

    def predict(self,train):
        x = []
        for i in self.estimators:
            if self.estimators_name[self.estimators.index(i)] != 'xgb'and self.estimators_name[self.estimators.index(i)] != 'lgbm':
                result = i.predict_proba(train)
                for j in range(3):
                    x.append(result[:,j])
            else:
                if self.estimators_name[self.estimators.index(i)]=='lgbm':
                    result = self.lgbm_model.predict(train, num_iteration=self.lgbm_model.best_iteration)
                    for j in range(3):
                        x.append(result[:,j])
                else:
                    result = self.xgb_model.predict(xgb.DMatrix(pd.DataFrame(train)))
                    for j in range(3):
                        x.append(result[:,j])
        x = np.array(x).T
        return self.model.predict(xgb.DMatrix(pd.DataFrame(x)))

    def score(self,x,y):
        # return metrics.log_loss(y,self.predict(x))
        print(self.clf.cv(num_boost_round=num_rounds_, early_stopping_rounds=50, params=list(param.items()),
                          dtrain=xgb.DMatrix(pd.DataFrame(self.predict(x)),pd.DataFrame(y))))
