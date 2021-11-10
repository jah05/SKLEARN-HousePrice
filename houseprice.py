import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import pickle

# pd.set_option("display.max_columns", None)

path = "data/"

def cleanData(data):
    missing_val_col = ['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st',
           'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond',
           'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
           'BsmtHalfBath', 'KitchenQual', 'Functional', 'FireplaceQu',
           'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
           'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
           'SaleType']          # found through test code (covers both test.csv and train.csv)

    # columns to onehot encode
    one_hot_col = ['MSSubClass', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'Electrical', 'PavedDrive', 'SaleCondition']

    # column maps
    LotShapeMap = {'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3}
    qS = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
    yn = {'Y':1, 'y':1, 'N':0, 'n':0}
    slope = {'Gtl':0, 'Mod':1, 'Sev':2}

    # good --> bad or bad --> good; (0-n-1) order doesn't matter because model can account
    maps = {'LotShape':LotShapeMap, 'LandSlope':slope, 'OverallQual':'sub1', 'OverallCond':'sub1', 'YearBuilt':'year', 'YearRemodAdd':'year', 'ExterQual':qS, 'ExterCond':qS, 'HeatingQC':qS, 'CentralAir':yn, 'YrSold':'year', 'MoSold':'sub1'}

    data = data.drop(missing_val_col, axis=1)            # drop columns with missing data

    numerical = set(data.columns) - set(one_hot_col) - set(maps.keys())     # take numerical cols

    new_data = pd.DataFrame(data[numerical])

    # apply maps and concatenate to new data
    for col in maps:
        temp_df = data[col]
        map = maps[col]
        if type(map) == dict:
            new_data[col] = temp_df.map(map)
        elif "sub" in map:
            new_data[col] = temp_df - int(map.split("sub")[-1])
        elif "year":
            new_data[col] = 2021 - temp_df

    # create one hot data
    for col in one_hot_col:
        ohc = OneHotEncoder()
        ohe = ohc.fit_transform(data[col].values.reshape(-1, 1)).toarray()
        col_names = ohc.get_feature_names(input_features=[col])
        temp_df = pd.DataFrame(ohe, columns=col_names)
        new_data = pd.concat([new_data, temp_df], axis=1)

    return new_data

if __name__ == '__main__':
    givenData = pd.read_csv(path+"train.csv")
    givenData= givenData.iloc[:, 1:]              # remove id
    y = givenData["SalePrice"]
    y /= 100000                 # scale prices so gradient doesn't get wrecked by loss
    givenData = givenData.drop("SalePrice", axis=1)

    print(givenData.shape)
    X = cleanData(givenData)
    print(X.shape)
    # correlation matrix
    fig = plt.figure()
    sns.heatmap(X.iloc[:,:31].corr())
    plt.show()

    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # standard normalization of data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=100)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # plot energy graph - want 95% preservation
    energy_graph = np.cumsum(pca.explained_variance_ratio_ * 100)
    plt.plot(energy_graph)
    plt.xlabel("Number of components (Dimensions)")
    plt.ylabel("Explained variance (%)")
    plt.show()

    model = SVR(kernel="linear")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(preds)
    print(y_test)

    mse = mean_squared_error(preds, y_test)
    print(mse)
    print(np.std(y_test))
