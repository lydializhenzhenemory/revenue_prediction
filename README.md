# Revenue Prediction Project

## Instruction 
Original dataset in `training_data_all_oem_code_sample.cvs`. 

In `1_preprocessing.py` (line 100-101) and `3_data_processing` (line 71-72) change the target column to the name of the column that trying to predict. Current code is using 'amt_revenue' as target column. 

`1_preprocessing.py`: 
```bash
# Assuming 'amt_revenue' is the target variable
correlation_matrix(normalized_df, df, 'normalized_corrMat.png', 'amt_revenue')
```

`3_data_processing`
```bash
# Specify the name of the target column, change to the target colum
target_column = 'amt_revenue'
```


Run file 1, 2, and 3 in order to process the original dataset. 

## Files
- `lr.py` for linear regression model 
- `neural_network.py` for neural network model 
- `rf.py` for random forest model 