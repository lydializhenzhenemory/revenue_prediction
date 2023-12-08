import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def clean_and_process_data(df):

    # # Data Type Conversion with specific date format if known
    # date_cols = ['period_input', 'period_fcst']
    # for col in date_cols:
    #     df[col] = pd.to_datetime(df[col], format='your_date_format_here', errors='coerce')

    # Removing Duplicates
    df.drop_duplicates(inplace=True)

    # Renaming Columns for clarity
    df.columns = [col.replace(' ', '_').lower() for col in df.columns]

    # Normalizing numerical columns (Ensure these columns exist)
    numerical_cols = ['qty_fgi_inventory', 'amt_fgi_inventory'] # Update with actual column names
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # # Feature Engineering for date columns
    # for col in date_cols:
    #     df[f'{col}_year'] = df[col].dt.year
    #     df[f'{col}_month'] = df[col].dt.month

    return df

def split_data(df, target_column):
    # Splitting the data into features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def select_relevant_features(df):
    # Define the features that are likely to have an impact on revenue

    # relevant_features = ['row_id', 'period_input', 'period_fcst', 'product_no_family', 
    #                       'model_no_family', 'customer_project_family', 'qty_fcst', 
    #                       'amt_revenue', 'qty_revenue', 'amt_cost_group', 'amt_fcst_max', 
    #                       'qty_fcst_max', 'amt_backlog', 'qty_backlog', 'qty_FGI_inventory', 
    #                       'amt_FGI_inventory', 'amt_budget', 'dt_mp', 'stat_endoflife', 
    #                       'main_parts_1', 'qty_onhand_main_parts_1', 'leadtime_main_parts_1', 
    #                       'main_parts_2', 'qty_onhand_main_parts_2', 'leadtime_main_parts_2', 
    #                       'main_parts_3', 'qty_onhand_main_parts_3', 'leadtime_main_parts_3',
    #                       'main_parts_4', 'qty_onhand_main_parts_4', 'leadtime_main_parts_4', 
    #                       'main_parts_5', 'qty_onhand_main_parts_5', 'leadtime_main_parts_5',]

    relevant_features = ['row_id', 'dt_input', 'period_input', 'dt_fcst', 'period_fcst', 'product_no', 
                         'product_no_family', 'model_no', 'model_no_family', 'customer_project', 
                         'customer_project_family', 'oem_code', 'qty_fcst', 'amount_fcst', 'amt_revenue', 
                         'qty_revenue', 'amt_cost_group', 'amt_fcst_min', 'amt_fcst_max', 'qty_fcst_min', 
                         'qty_fcst_max', 'amt_backlog', 'qty_backlog', 'qty_FGI_inventory', 'amt_FGI_inventory', 
                         'amt_projection', 'amt_budget', 'stage_project', 'dt_mp', 'month_mp', 'stat_endoflife', 
                         'dt_endoflife', 'main_parts_1', 'main_parts_category_1', 'qty_onhand_main_parts_1', 
                         'receive_main_parts_1', 'leadtime_main_parts_1', 'leadtime_main_parts_his_1', 'main_parts_2', 
                         'main_parts_category_2', 'qty_onhand_main_parts_2', 'receive_main_parts_2', 
                         'leadtime_main_parts_2', 'leadtime_main_parts_his_2', 'main_parts_3', 'main_parts_category_3', 
                         'qty_onhand_main_parts_3', 'receive_main_parts_3', 'leadtime_main_parts_3', 
                         'leadtime_main_parts_his_3', 'main_parts_4', 'main_parts_category_4', 'qty_onhand_main_parts_4', 
                         'receive_main_parts_4', 'leadtime_main_parts_4', 'leadtime_main_parts_his_4', 'main_parts_5', 
                         'main_parts_category_5', 'qty_onhand_main_parts_5', 'receive_main_parts_5', 'leadtime_main_parts_5', 
                         'leadtime_main_parts_his_5']

    # Selecting only the relevant features from the dataframe
    selected_df = df[relevant_features]

    return selected_df

# Load the dataset
file_path = 'imputed_normalized_data.csv'
data = pd.read_csv(file_path)

# # Feature Selection
# selected_features = select_relevant_features(data)
# data = select_relevant_features(data)

# Specify the name of your target column, change to the target colum
target_column = 'amt_revenue'

# Splitting the data
X_train, X_test, y_train, y_test = split_data(data, target_column)

# Optionally, save the split datasets
X_train.to_csv('xTrain.csv', index=False)
X_test.to_csv('xTest.csv', index=False)
y_train.to_csv('yTrain.csv', index=False)
y_test.to_csv('yTest.csv', index=False)
