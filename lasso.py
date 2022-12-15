from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy
import scipy.stats as stats
from statistic import dataset

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
pd.set_option("display.expand_frame_repr",False)
data = dataset
print('数据展示：',data)
print('数据形状：',data.shape)
print('\n数据示例：')
data.head(10)
print("各样本统计量描述")
print(data.describe())

# ols
X = data.copy()
X = X.drop(['date'],axis=1)
print('X:',X)
# standardize the data
# scaler = StandardScaler()
#X[['income_of_centre_government','income_of_local_goverment','debt_of_local_goverment','GDP','PMI','Consumer_Confidence_Index','CPI','PPI','SHIBOR','Treasury_Bill_Rate','exchange_rate','stock_dataset_per_season']] = scaler.fit_transform(X[['income_of_centre_government','income_of_local_goverment','debt_of_local_goverment','GDP','PMI','Consumer_Confidence_Index','CPI','PPI','SHIBOR','Treasury_Bill_Rate','exchange_rate']])=scaler.fit_transform(X[['income_of_centre_government','income_of_local_goverment','debt_of_local_goverment','GDP','PMI','Consumer_Confidence_Index','CPI','PPI','SHIBOR','Treasury_Bill_Rate','exchange_rate','stock_dataset_per_season']])
# print('standard X:',X)

formula = 'stock_dataset_per_season ~ income_of_centre_goverment + income_of_local_goverment + debt_of_local_goverment + GDP + PMI + Consumer_Confidence_Index + CPI + PPI + SHIBOR + Treasury_Bill_Rate + exchange_rate'
model = smf.ols(formula, data=data).fit()
print(model.summary())

# lasso regression
X1 = data.copy()
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from sklearn.metrics import mean_squared_error
predictors = X1.drop(['date','stock_dataset_per_season'],axis=1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(predictors, X1['stock_dataset_per_season'], test_size=0.3, random_state=0)
Lambda = np.logspace(-5, 10, 200)
lasso_cv = LassoCV(alphas=Lambda, normalize=False, cv=10, max_iter=100000)
lasso_cv.fit(x_train, y_train)
print('最佳惩罚系数:',lasso_cv.alpha_)

lasso = Lasso(alpha=lasso_cv.alpha_, normalize=False, max_iter=100000)
lasso.fit(x_train, y_train)

print('系数列表',pd.DataFrame(index=['intercept']+list(x_train.columns),data=np.append(lasso.intercept_,lasso.coef_)))

# model evaluation
lasso_pred = lasso.predict(x_test)
MSE = mean_squared_error(y_test, lasso_pred)
print('MSE:',MSE,'\n最优lambda:',lasso_cv.alpha_)

# plot
import matplotlib.pyplot as plt
from sklearn import linear_model

x2 = x_train
y2 = y_train
x3 = np.array(x2)
y3 = np.array(y2)

# using Lars path to find the best lambda
# This differs from a coordinate descent based implementation, which produces an exact solution as a segmented linear function of the parametrization of its coefficients.using
_, n3, coefs = linear_model.lars_path(x3, y3, method='lasso', verbose=True)
print("进入变量的自变量索引值：", n3)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /=xx[-1]
plt.figure(figsize=(10, 10))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('Lasso Path')
plt.axis('tight')
plt.legend(np.array(predictors)[n3], fontsize=4, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
print("前四个进入模型的自变量：", np.array(predictors)[n3[:4]])
plt.savefig('lasso_path正则化.png')
plt.show()

