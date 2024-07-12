import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load data
data_train = pd.read_csv('train_data.csv')

#data_predict = pd.read_csv('predict_data.csv')

# data processing
data_train.fillna(data_train.mean, inplace=True)
data_train.drop_duplicates(inplace=True)


data_train['date'] = pd.to_datetime(data_train['date'])
data_train = pd.get_dummies(data_train, columns=['city', 'statezip'])


data_train['age'] = 2024 - data_train['yr_built']
data_train['renovated'] = data_train['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)


scaler = StandardScaler()

numeric_features = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'age', 'renovated']

data_train[numeric_features] = scaler.fit_transform(data_train[numeric_features])

# drop non-numeric columns
data_train = data_train.drop(columns=['street', 'country', 'date'])  # burayı droplayıp 35.satırda data.corr() yapmayı denedim ama onda da kötü bir çıktı verdi
data_train.describe()


# plot histograms
plt.figure(figsize=(12, 8))
data_train.hist(bins=20)
plt.tight_layout()
plt.show()

# correlation matrix
corr_matrix = data_train.corr()

# plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# separate data into test and training
X = data_train.drop('price', axis=1)
y = data_train['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# start model
tree_reg = DecisionTreeRegressor()

# train model
tree_reg.fit(X_train, y_train)


# Model evaluation function
def evaluate_model(model, X_test, y_test):
    predicts = model.predict(X_test)
    mse = mean_squared_error(y_test, predicts)
    r2 = r2_score(y_test, predicts)
    return mse, r2


tree_reg_mse, tree_reg_r2 = evaluate_model(tree_reg, X_test, y_test)

print(f'Decision Tree Regression MSE: {tree_reg_mse}, R-square: {tree_reg_r2}')

