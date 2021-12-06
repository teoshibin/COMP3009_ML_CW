# Artificial Neural Networks

## Datasets
- Classsification
    [Heart failure clinical records Data Set](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
    
- Regression
    [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

## Folder Structure
```
    .
    └── cw3/
        ├── datasets/
        ├── results/
        ├── functions/
        ├── cv_classification.py
        ├── cv_regression.py
        └── REQUIREMENTS.md
```
`cv_classification.py` show the evaluation of artificial neural network on the classification task.  
`cv_regression.py` show the evaluation of artificial neural network on the regression task.    
`functions` contain all functions related to the artificial neural network.   
`REQUIREMENTS.md` show all the required packages version

## Dependencies Related
- [Requirements](REQUIREMENTS.md)
- [Anaconda Commands Doc](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## VScode configuration
```batch
    conda activate mle_tf
    code .
```
navigate into your project folder, 
select intepreter from `mle_tf` env then save this workspace