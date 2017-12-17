import ensemble
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
import lightgbm as lgbm

data_combine = pd.read_excel('D:/大学/大三上/数据挖掘/作业/final_data.xlsx')
train_data = data_combine[:-8392]
print(train_data.describe())
test_data = data_combine[-8392:]
test_data = test_data.drop('author',axis=1)
train_x = train_data.drop('author',axis=1).values
train_y = train_data['author'].values
print('数据载入完成')


bcf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=200)
rf = RandomForestClassifier(n_estimators=200,max_features=30)
etc = ExtraTreesClassifier(n_estimators=300,max_features=30)
gbc = GradientBoostingClassifier(n_estimators=100)
print('模型初始化完成')

kf = KFold(n_splits=5,shuffle=True,random_state=2017)
bag = ensemble.ensemble([('lgbm',lgbm),('xgb',xgb),('etc',etc),('gbc',gbc)])
a = []

# for val_index,dva_index in kf.split(train_x):
#     x_val,x_dva = train_x[val_index],train_x[dva_index]
#     y_val,y_dva = train_y[val_index],train_y[dva_index]
#     bag.fit(x_val,y_val)
#     a.append(bag.score(x_dva,y_dva))
#     print(a)
# print(a)
bag.fit(train_x,train_y)
a = pd.DataFrame(bag.predict(test_data.values))
print('正在保存')
a.to_csv('d:/final_data.csv')

