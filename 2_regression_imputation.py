
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np

def regression_imputation(df):
    # Identifying columns with missing values
    cols_with_missing_values = df.columns[df.isnull().any()].tolist()
    
    for col in cols_with_missing_values:
        # Using other columns to train a regression model
        # Creating a new dataframe without the target column
        df_without_target = df.drop(col, axis=1)
        
        # Splitting the dataset into rows with and without missing values in the target column
        df_with_values = df_without_target[~df[col].isnull()]
        df_missing_values = df_without_target[df[col].isnull()]
        
        # Extracting the target values
        target = df[col].dropna()

        # If there are no rows to train the model, we fall back to mean imputation
        if len(df_with_values) == 0:
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[[col]])
        else:
            # Training the regression model
            model = LinearRegression()
            model.fit(df_with_values, target)
            
            # Predicting missing values
            predicted_values = model.predict(df_missing_values)
            
            # Replacing missing values with predicted values
            df.loc[df[col].isnull(), col] = predicted_values
    
    return df

# Retry the process of applying regression imputation and saving the imputed dataset
# Load the dataset again
df_retry = pd.read_csv('normalized_data.csv')

# Applying the regression imputation function to the dataset
df_imputed_retry = regression_imputation(df_retry)

# Save the imputed dataset to a new file
output_file_path_retry = 'imputed_normalized_data.csv'
df_imputed_retry.to_csv(output_file_path_retry, index=False)

output_file_path_retry

