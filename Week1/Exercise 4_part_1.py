#EXERCISE 4 _(1)
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#load iris data from datasets
iris = datasets.load_iris()
x1,x2,x3,x4 = [iris.data[:,np.newaxis, col] for col in range(iris.data.shape[1])]

#function to compute r2 score for feature X and y
def return_r2_score(x,y):
    lm = LinearRegression()
    lm.fit(x,y)
    y_predict = lm.predict(x)
    return r2_score(y, y_predict)

#run r2 scores for respective feature combinations from data
def run():
    print("""The r2 scores:
          feature 1 vs feature 3: {:.4f}
          feature 1 vs feature 2: {:.4f}
          feature 1 vs feature 4: {:.4f}""".format(return_r2_score(x3, x1),\
          return_r2_score(x2, x1), return_r2_score(x4, x1)))
run() #execute for results

# outputs
# feature 1 vs feature 3: 0.7600
# feature 1 vs feature 2: 0.0138
# feature 1 vs feature 4: 0.6690
#Conclusion: feature 3 explains more variation in feature1 than features 2 and 4   