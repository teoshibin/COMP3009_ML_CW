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
        │   ├── data_IO.py
        │   ├── data_preprocessing.py
        │   ├── data_splitting.py
        │   ├── math.py
        │   ├── metrics.py
        │   ├── neural_network.py
        │   └── plots.py
        ├── cv_classification.py
        ├── cv_regression.py
        ├── REQUIREMENTS.md
        └── README.md
```


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