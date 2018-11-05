
#######    Ridge vs Lasso Regression #########

#   Ridge Regression  ######

import pandas as pd
import numpy as np
adver = pd.read_csv('data/adver.csv', index_col=0)
x = np.array(adver[['TV', 'Radio', 'Newspaper']])
y = np.array(adver['Sales'])

from sklearn import linear_model
ridge = linear_model.Ridge(alpha = 1) # create a ridge regression instance
ridge.fit(x, y) # fit data
ridge.coef_, ridge.intercept_ # print out the coefficients

print("The determination of ridge regression is: %.4f" %ridge.score(x, y))

## prediction 
ridge.predict([[5.2, 18.5, 3.7]])

from sklearn import linear_model
ridge = linear_model.Ridge()

alpha_100 = np.logspace(0, 8, 100)
coef = []
for i in alpha_100:
    ridge.set_params(alpha = i)
    ridge.fit(x, y)
    coef.append(ridge.coef_)

df_coef = pd.DataFrame(coef, index=alpha_100, columns=['TV', 'Radio', 'Newspaper'])
import matplotlib.pyplot as plt
title = 'Ridge coefficients as a function of the regularization'
axes = df_coef.plot(logx=True, title=title)
axes.set_xlabel('alpha')
axes.set_ylabel('coefficients')
plt.show()


###### Lasso  Regression  ########

## coefficients
lasso = linear_model.Lasso(alpha=1) # create a lasso instance
lasso.fit(x, y) # fit data
lasso.coef_, lasso.intercept_ # print out the coefficients

## determination
print("The determination of ridge regression is: %.4f" %lasso.score(x, y))

## prediction
lasso.predict([[5.2, 18.5, 3.7]])

alphas_lasso = np.logspace(-2, 4, 100)
coef_lasso = []

for i in alphas_lasso:
    lasso.set_params(alpha=i).fit(x, y)
    coef_lasso.append(lasso.coef_)

columns = ['TV', 'Radio', 'Newspaper']
df_coef = pd.DataFrame(coef_lasso, index=alphas_lasso, columns=columns)
title = 'Lasso coefficients as a function of the regularization'
df_coef.plot(logx=True, title=title)
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.show()

