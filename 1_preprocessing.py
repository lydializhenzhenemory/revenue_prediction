import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def cal_corr(df, file_name):
    corrMat = df.corr()
    plt.figure(figsize=(10,8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corrMat, cmap=cmap, center=0, square=True, annot=False,
                linewidths=.5, cbar_kws={"shrink": 0.75},
                annot_kws={"fontsize": 5, "ha": 'center', 'va': 'center'})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(file_name)
    # Close the plot to release resources
    plt.close()
    return corrMat


def correlation_matrix(x, y, file_name, y_col_name):
    new_x = x.copy()
    new_x['target'] = y[y_col_name]

    # Convert date strings to datetime objects and then to ordinal numbers
    if 'period_date' in new_x.columns:
        new_x['period_date'] = pd.to_datetime(new_x['period_date']).apply(lambda date: date.toordinal())

    # Filter out non-numeric columns
    numeric_cols = new_x.select_dtypes(include=[np.number]).columns.tolist()
    new_x = new_x[numeric_cols]

    cal_corr(new_x, file_name)


def normalize_data(df):
    # Select only numeric columns
    # df = df.select_dtypes(include=[np.number])

    # Define features to normalize based on your dataset
    features_to_normalize = ['amt_cost_group', 'qty_revenue', 'amt_revenue',
                             'qty_fcst_avg', 'qty_fcst_max', 'qty_fcst_min',
                             'amt_projection', 'amt_budget', 'qty_inv_FGI']

    # Only normalize columns that exist in the DataFrame
    existing_features = [feature for feature in features_to_normalize if feature in df.columns]
    
    scaler = StandardScaler()
    if existing_features:  # Check if there are features to normalize
        df[existing_features] = scaler.fit_transform(df[existing_features])

    # Handle categorical data
    categorical_data = ['dt_input', 'dt_fcst', 'product_no', 'product_no_family',
                        'model_no', 'model_no_family', 'customer_project', 'customer_project_family',
                        'oem_code', 'stage_project', 'dt_mp', 'stat_endoflife', 'dt_endoflife', 
                        'main_parts_1', 'main_parts_category_1', 'main_parts_2', 
                        'main_parts_category_2', 'main_parts_3', 'main_parts_category_3', 
                        'main_parts_4', 'main_parts_category_4', 'main_parts_5', 'main_parts_category_5']
    
    # Loop through each categorical column and apply Label Encoding
    for col in categorical_data:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to ensure LabelEncoder works

    # Handling Outliers after normalization
    for feature in existing_features:
        # Calculate IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Apply capping
        df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
        df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])

    # Other pre-processing steps from the previous version of the function if necessary
    # ...

    # Ensure only numerical data is returned
    df = df.select_dtypes(include=[np.number])
    
    return df


def main():
    # Load your dataset
    df = pd.read_csv('training_data_all_oem_code_sample.csv')

    # Handling Missing Values
    df.fillna(0, inplace=True)
    
    # Normalize the data
    normalized_df = normalize_data(df)

    # Save the processed (normalized) data to a CSV file
    normalized_df.to_csv('normalized_data.csv', index=False)

    # Assuming 'amt_revenue' is your target variable
    correlation_matrix(normalized_df, df, 'normalized_corrMat.png', 'amt_revenue')

if __name__ == "__main__":
    main()