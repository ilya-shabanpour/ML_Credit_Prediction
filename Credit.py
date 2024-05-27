import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor


def replace_outliers_with_mode_iqr(data):
    # Initialize an empty DataFrame to store the results
    result = pd.DataFrame(index=data.index, columns=data.columns)

    # Loop through each column in the DataFrame
    for col in data.columns:
        Q1 = data[col].quantile(0.25)  # 25th percentile (Q1)
        Q3 = data[col].quantile(0.75)  # 75th percentile (Q3)
        IQR = Q3 - Q1  # Interquartile Range (IQR)
        lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
        upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers

        # Replace outliers with mode for the current column
        mode_val = data.loc[(data[col] >= lower_bound) & (data[col] <= upper_bound), col].mode()[0]
        result[col] = data[col].apply(lambda x: mode_val if x < lower_bound or x > upper_bound else x)

    return result

def data_preparation(df):
    """""
    data preprocessing

    """""

    # drop last column because all NaN
    df = df.drop("Unnamed: 19", axis=1)
    df = df.drop("CLIENTNUM", axis=1)

    # replace mod of Marital Status for NaN and Unknown
    mod_marriage = df["Marital_Status"].mode()
    mod_marriage = mod_marriage[0]
    new_col = df["Marital_Status"].replace(np.nan, mod_marriage)
    df["Marital_Status"] = new_col.values
    new_col = df["Marital_Status"].replace("Unknown", mod_marriage)
    df["Marital_Status"] = new_col.values

    # replace mod of Gender for NaN
    mod_gender = df["Gender"].mode()
    mod_gender = mod_gender[0]
    new_col = df["Gender"].replace(np.nan, mod_gender)
    df["Gender"] = new_col.values

    # replace mod of Education_Level for Unknown
    mod_edu = df["Education_Level"].mode()
    mod_edu = mod_edu[0]
    new_col = df["Education_Level"].replace("Unknown", mod_edu)
    df["Education_Level"] = new_col.values

    # replace mod Income_Category for Unknown
    mod_inc = df["Income_Category"].mode()
    mod_inc = mod_inc[0]
    new_col = df["Income_Category"].replace("Unknown", mod_inc)
    df["Income_Category"] = new_col.values

    # replace mod Card_Category for NaN
    mod_card_cat = df["Card_Category"].mode()
    mod_card_cat = mod_card_cat[0]
    new_col = df["Card_Category"].replace(np.nan, mod_card_cat)
    df["Card_Category"] = new_col.values

    # replace mean Months_on_book for NaN
    mean_month = df["Months_on_book"].mean()
    mean_month = mean_month.round()
    new_col = df["Months_on_book"].replace(np.nan, mean_month)
    df["Months_on_book"] = new_col.values

    # replace mean Total_Relationship_count for NaN
    mean_total = df["Total_Relationship_Count"].mean()
    mean_total = mean_total.round()
    new_col = df["Total_Relationship_Count"].replace(np.nan, mean_total)
    df["Total_Relationship_Count"] = new_col.values

    # Now we convert the string type columns into numbers using One Hot Encoding

    df_encoded = pd.get_dummies(df['Marital_Status'], prefix='Marital_Status')
    df_encoded = df_encoded.astype(int)
    df = pd.concat([df, df_encoded], axis=1)
    df.drop('Marital_Status', axis=1, inplace=True)

    df_encoded = pd.get_dummies(df['Card_Category'], prefix='Card_Category')
    df_encoded = df_encoded.astype(int)
    df = pd.concat([df, df_encoded], axis=1)
    df.drop('Card_Category', axis=1, inplace=True)

    df_encoded = pd.get_dummies(df['Gender'], prefix='Gender')
    df_encoded = df_encoded.astype(int)
    df = pd.concat([df, df_encoded], axis=1)
    df.drop('Gender', axis=1, inplace=True)

    df_encoded = pd.get_dummies(df["Education_Level"], prefix="Education_Level")
    df_encoded = df_encoded.astype(int)
    df = pd.concat([df, df_encoded], axis=1)
    df.drop("Education_Level", axis=1, inplace=True)

    df_encoded = pd.get_dummies(df["Income_Category"], prefix="Income_Category")
    df_encoded = df_encoded.astype(int)
    df = pd.concat([df, df_encoded], axis=1)
    df.drop("Income_Category", axis=1, inplace=True)

    # move the label column to be the last column
    column_to_move = df.pop("Credit_Limit")
    df.insert(len(df.columns), "Credit_Limit", column_to_move)
    return df


def model(df):
    pf = PolynomialFeatures(degree=2)

    scaler = StandardScaler()
    # scaler = MinMaxScaler()

    y = df.pop("Credit_Limit")
    # df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    x = np.array(df)

    df.insert(len(df.columns), "Credit_Limit", y)

    y = np.array(y)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

    # x_train_poly = pf.fit_transform(x_train)
    # x_test_poly = pf.transform(x_test)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # x_train_poly = scaler.fit_transform(x_train_poly)
    # x_test_poly = scaler.transform(x_test_poly)

    # Model = LinearRegression()
    # Model = Ridge()
    Model = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=15)
    # Model = MLPRegressor(max_iter=1500, random_state=0, alpha= 0.0001, learning_rate='adaptive', learning_rate_init=0.01, hidden_layer_sizes=(100,), solver='adam')

    Model.fit(x_train, y_train)
    y_pred = Model.predict(x_test)

    # Model.fit(x_train_poly, y_train)
    # y_pred = Model.predict(x_test_poly)

    mse = round(mean_squared_error(y_true=y_test, y_pred=y_pred))
    # r2 = round(r2_score(y_test, y_pred), 2)
    # mae = round(mean_absolute_error(y_test, y_pred))
    rmse = round(math.sqrt(mse))

    # print(r2)
    print(mse)
    print(rmse)
    return [mse, rmse]


if __name__ == '__main__':
    df = pd.read_csv("creditPrediction.csv")

    df = data_preparation(df)

    df = replace_outliers_with_mode_iqr(df)

    results =[]
    for i in range(10):
        result = model(df)
        results.append(result)
    print(results)