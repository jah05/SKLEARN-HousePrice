import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

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


givenData = pd.read_csv(path+"train.csv")
givenData= givenData.iloc[:, 1:]              # remove id
prices = givenData["SalePrice"]
givenData = givenData.drop("SalePrice", axis=1)
cleanedData = cleanData(givenData)
