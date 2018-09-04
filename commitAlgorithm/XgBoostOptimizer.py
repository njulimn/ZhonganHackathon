
# coding: utf-8

# In[27]:


#@Team: 生男孩48
#@Author: tenk
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn import cross_validation
from scipy import stats
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


# In[92]:


#step1: 数据读取 + 特征抽取
##数据目录读取抽取的Feature后的训练数据文件 data.csv
data =  np.loadtxt('../data/data.csv',delimiter=',')

##  X对应特征向量 Y数据标签，
X = data[:,:-1]
Y = data[:,-1]

#@  使用随机决策树进行特征选择
clf = ExtraTreesClassifier()
clf = clf.fit(X, Y)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
X = model.transform(X)


# In[36]:


#step2: xgboost超参数求解
#step2.1: 固定其他参数，求解最优subsample,max_delta_step参数
cv_params = {'subsample': [0.8,0.85,0.9,0.95], 'max_delta_step': [1,2,3,4]}
fix_params = {'learning_rate': 0.2, 'n_estimators': 100, 'objective': 'binary:logistic', 'max_depth': 6, 'min_child_weight':1}
csv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring = 'f1', cv = 5) 
csv.fit(X, Y)
csv.grid_scores_
## 输出最优subsample,max_delta_step参数
csv.best_params_


# In[39]:


#step2.2: 固定其他参数，求解最优learning_rate参数
cv_params = {'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}
fix_params['max_delta_step'] = 1
fix_params['subsample'] = 0.95
csv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring = 'f1', cv = 5) 
csv.fit(X, Y)
csv.grid_scores_
##  输出最优learning_rate
csv.best_params_
## 求解得到最优参数
fix_params['learning_rate'] = 0.05
params_final =  fix_params


# In[72]:


#step3: 十次交叉验证算法效果
numFolds = 10
folds = cross_validation.KFold(n = len(X), shuffle = True, n_folds = numFolds)
estimators = []
results = np.zeros(len(X))
score = []
for train_index, test_index in folds:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    xgdmat_train = xgb.DMatrix(X_train, y_train)
    xgdmat_test = xgb.DMatrix(X_test, y_test)
    xgb_final = xgb.train(params_final, xgdmat_train, num_boost_round = 100)
    thresh = 0.38
    y_pred = xgb_final.predict(xgdmat_test)
    y_pred [y_pred > thresh] = 1
    y_pred [y_pred <= thresh] = 0
    score.append(f1_score(y_test, y_pred))
# 输出交叉验证得分
print(score)


# In[88]:


#step4: 读取测试集特征向量，预测结果并输出
re_np = np.loadtxt('../data/test.csv',delimiter=',')
xgdmat_test = xgb.DMatrix(re_np)
label = xgb_final.predict(xgdmat_test)
thresh = 0.39
label [label > thresh] = 1
label [label <= thresh] = 0
print(len([x for x in label if x==0]))
print(label.shape)
## 读入测试数据，输出到文件data目录下out文件
infile = open('../data/result.csv')
outfile = open('../data/out.csv','w')
#write to file
cnt = 0
for line in infile:
    if cnt < 1:
        outfile.write(line.rstrip()+',label'+'\n')
    else: 
        outfile.write(line.rstrip()+',')
        outfile.write(str(int(label[cnt-1]))+'\n')
    cnt += 1

