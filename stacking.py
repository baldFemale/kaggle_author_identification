import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
import lightgbm as lgbm
import numpy as np

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

param_lgmb = {}
param_lgmb['objective']='multiclass'
param_lgmb['num_class']=3
param_lgmb['metric']='multi_logloss'
param_lgmb['learning_rate']=0.1
param_lgmb['num_leaves']=15
param_lgmb['max_depth']=5
param_lgmb['seed']=2017

data_combine = pd.read_excel('D:/大学/大三上/数据挖掘/作业/final_data.xlsx')
train_data = data_combine[:-8392]
print(train_data.describe())
test_data = data_combine[-8392:]
test_data = test_data.drop('author',axis=1).values
train_x = train_data.drop('author',axis=1).values
train_y = train_data['author'].values
print('数据载入完成')

bcf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=200)
rf = RandomForestClassifier(n_estimators=200,max_features=30)
etc = ExtraTreesClassifier(n_estimators=300,max_features=30)
gbc = GradientBoostingClassifier(n_estimators=100)
bag = [etc,gbc,rf]
bag_all = [lgbm,xgb,etc,gbc,rf]
# bag_all = [lgbm]
# bag_all_name = ['lgbm']
bag_all_name = ['lgbm','xgb','etc','gbc','rf']
print('模型初始化完成')

def stack(train_x,train_y,test_x,test_y,test_data,clf):
    if clf in bag:
        clf.fit(train_x,train_y)
        pred_train = clf.predict_proba(test_x)
        pred_test = clf.predict_proba(test_data)
    else:
        if clf==xgb:
            model = clf.train(dtrain=xgb.DMatrix(pd.DataFrame(train_x),pd.DataFrame(train_y)),num_boost_round=500,params=list(param.items()))
            pred_train = model.predict(xgb.DMatrix(pd.DataFrame(test_x)))
            pred_test = model.predict(xgb.DMatrix(pd.DataFrame(test_data)))
        else:
            lgbm_train = lgbm.Dataset(train_x,train_y)
            lgbm_test = lgbm.Dataset(test_x, test_y, reference=lgbm_train)
            model = clf.train(params=param_lgmb,train_set=lgbm_train,valid_sets=lgbm_test,num_boost_round=500,early_stopping_rounds=20)
            pred_train = model.predict(test_x,num_iteration=model.best_iteration)
            pred_test = model.predict(test_data,num_iteration=model.best_iteration)
    return pred_train, pred_test

kf = KFold(n_splits=5,shuffle=True,random_state=2017)
data_train = pd.DataFrame()
data_test = pd.DataFrame()
for i in bag_all:
    cv = []
    pred_full_test = np.zeros([len(test_data),3])
    pred_full_train = np.zeros([len(train_x),3])
    for val_index,dva_index in kf.split(train_x):
        x_val,x_dva = train_x[val_index],train_x[dva_index]
        y_val,y_dva = train_y[val_index],train_y[dva_index]
        pred_train,pred_test = stack(x_val,y_val,x_dva,y_dva,test_data,i)
        pred_full_test+=pred_test
        pred_full_train[dva_index,:] = pred_train
        cv.append(metrics.log_loss(y_dva,pred_train))
    print(cv)
    pred_full_test/=5
    for j in range(3):
        data_train[bag_all_name[bag_all.index(i)]+str(j)] = pred_full_train[:,j]
        data_test[bag_all_name[bag_all.index(i)]+str(j)] = pred_full_test[:,j]

result = pd.concat([data_train,data_test],axis=0)
result.to_csv('d:/result.csv')