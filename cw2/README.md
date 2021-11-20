# Decision Tree


## Datasets
- Classsification
    [Heart failure clinical records Data Set](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
    
- Regression
    [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

## Folder Structure

```
    .
    └── cw2/
        ├── datasets/
        ├── functions/
        │   ├── data_processing/
        │   ├── decision_tree/
        │   ├── metrics/
        └── demo.m
        └── cross_validation_classification.m
        └── cross_validation_regression.m
```
`demo.m` show both tree are working for the datasets.
`cross_validation_classification.m` show the evaluation of decision tree on the classification task.
`cross_validation_regression.m` show the evalutation of decision tree on the regression task.
`data_processing` for data validation split and normalisation   
`decision_tree` for all decision tree related functions   
`metrics` for accuracy, f1, confusion matrix etc.   
