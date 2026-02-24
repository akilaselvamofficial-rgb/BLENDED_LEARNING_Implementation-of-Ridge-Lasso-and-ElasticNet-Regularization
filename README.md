# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, preprocess it by encoding categorical variables, separating features and target price, and scaling the data before splitting into training and testing sets.

2.Define Ridge, Lasso, and ElasticNet models and apply Polynomial Features using a pipeline.

3.Train each model on the training data and make predictions on the test data.

4.Compute MSE, MAE, and R² values for each model and display the comparison using bar charts. 

## Program:
```
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
df=pd.read_csv('encoded_car_data (1).csv')
df.head()
df=pd.get_dummies(df,drop_first=True)
X=df.drop('price',axis=1)
y=df['price']
scaler=StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(y.values.reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
models={
    "Ridge":Ridge(alpha=1.0),
    "Lasso":Lasso(alpha=1.0),
    "ElasticNet":ElasticNet(alpha=1.0,l1_ratio=0.5)
}
results={}
for name,model in models.items():
    pipeline=Pipeline([
        ('poly',PolynomialFeatures(degree=2)),
    ('regressor',model)
    ])
    pipeline.fit(X_train,y_train)
    predictions=pipeline.predict(X_test)
mse=mean_squared_error(y_test,predictions)
mae=mean_absolute_error(y_test,predictions)
r2=r2_score(y_test,predictions)
results[name]={'MSE':mse,'MAE':mae,'R2 Score':r2}
print('Name: AKILA S')
print('Reg. No:212225220008')
for model_name,metrics in results.items():
    print(f"{model_name} -Mean Squared Error: {metrics['MSE']:.2f},R2 Score: {metrics['R2 Score']:.2f},Mean Absolute Error: {metrics['MAE']:.2f}")
results_df=pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'},inplace=True)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.barplot(x='Model',y='MSE',data=results_df,palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.subplot(1,2,2)
sns.barplot(x='Model',y='R2 Score',data=results_df,palette='viridis')
plt.title('R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
Developed by: AKILA S
RegisterNumber: 212225220008 
*/
```

## Output:

<img width="809" height="89" alt="Screenshot 2026-02-24 132346" src="https://github.com/user-attachments/assets/fed3f352-702b-4221-8ef4-9d069056727c" />


<img width="356" height="84" alt="Screenshot 2026-02-24 132442" src="https://github.com/user-attachments/assets/17d4ad5b-3bdd-4411-a8bf-263310c32af0" />






<img width="441" height="670" alt="Screenshot 2026-02-24 132519" src="https://github.com/user-attachments/assets/49819e3d-00b3-465c-a0aa-c1d7905bc20f" />
<img width="528" height="607" alt="Screenshot 2026-02-24 132534" src="https://github.com/user-attachments/assets/e0ad7e21-0db9-4a7b-8135-de314284b7a5" />





## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
