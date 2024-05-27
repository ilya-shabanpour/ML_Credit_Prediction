import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor


def replace_outliers_with_mean_iqr(data):
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
        mean_val = data.loc[(data[col] >= lower_bound) & (data[col] <= upper_bound), col].mean()
        result[col] = data[col].apply(lambda x: mean_val if x < lower_bound or x > upper_bound else x)

    return result

def data_preparation(df):
    """""
    data preprocessing

    """""

    # drop last column because all NaN
    df.drop("Unnamed: 19", axis=1, inplace=True)
    df.drop("CLIENTNUM", axis=1, inplace=True)

    # replace mod of Marital Status for NaN and Unknown
    mod_marriage = df["Marital_Status"].mode()[0]
    new_col = df["Marital_Status"].replace(np.nan, mod_marriage)
    df["Marital_Status"] = new_col.values
    new_col = df["Marital_Status"].replace("Unknown", mod_marriage)
    df["Marital_Status"] = new_col.values

    # replace mod of Gender for NaN
    mod_gender = df["Gender"].mode()[0]
    new_col = df["Gender"].replace(np.nan, mod_gender)
    df["Gender"] = new_col.values

    # replace mod of Education_Level for Unknown
    mod_edu = df["Education_Level"].mode()[0]
    new_col = df["Education_Level"].replace("Unknown", mod_edu)
    df["Education_Level"] = new_col.values

    # replace mod Income_Category for Unknown
    mod_inc = df["Income_Category"].mode()[0]
    new_col = df["Income_Category"].replace("Unknown", mod_inc)
    df["Income_Category"] = new_col.values

    # replace mod Card_Category for NaN
    mod_card_cat = df["Card_Category"].mode()[0]
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

    df = df.drop_duplicates()

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

    x_train_poly = pf.fit_transform(x_train)
    x_test_poly = pf.transform(x_test)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_poly = scaler.fit_transform(x_train_poly)
    x_test_poly = scaler.transform(x_test_poly)

    lr = LinearRegression()
    ridge = Ridge()
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=10)
    # mlp = MLPRegressor(max_iter=1500, random_state=0, alpha= 0.0001, learning_rate='adaptive', learning_rate_init=0.01, hidden_layer_sizes=(100,), solver='adam')

    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)

    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)

    ridge.fit(x_train_poly, y_train)
    y_pred_ridge = ridge.predict(x_test_poly)

    mse_rf = round(mean_squared_error(y_true=y_test, y_pred=y_pred_rf))
    mse_lr = round(mean_squared_error(y_true=y_test, y_pred=y_pred_lr))
    mse_ridge = round(mean_squared_error(y_true=y_test, y_pred=y_pred_ridge))

    return mse_rf, mse_lr, mse_ridge


if __name__ == '__main__':
    df = pd.read_csv("creditPrediction.csv")

    df = data_preparation(df)

    df = replace_outliers_with_mean_iqr(df)

    results_RF = []
    results_LR = []
    results_PolyRidge = []
    for i in range(10):
        result = model(df)
        results_RF.append(result[0])
        results_LR.append(result[1])
        results_PolyRidge.append(result[2])


    results_RF = np.array(results_RF)
    results_LR = np.array(results_LR)
    results_PolyRidge = np.array(results_PolyRidge)

    plt.plot(results_LR, color='blue', label='Linear Regression - Mean: ' +  str(results_LR.mean()))

    plt.plot(results_PolyRidge, color='red', label='Polynomial Ridge Regression - Mean: ' +
                                                   str(results_PolyRidge.mean()))

    plt.plot(results_RF, color='green', label='Random Forest - Mean: ' +  str(results_RF.mean()))

    plt.title("3 Models MSE")
    plt.xlabel("Execution number")
    plt.ylabel("MSE")
    plt.grid()
    plt.legend()
    plt.show()

    print("Linear Regression mean MSE: ", results_LR.mean())
    print("Poly Ridge Regression mean MSE: ", results_PolyRidge.mean())
    print("Random Forest mean MSE: ", results_RF.mean())
