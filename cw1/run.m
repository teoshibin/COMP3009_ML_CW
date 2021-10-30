% directory should be in cw1

close all
clear
clc

addpath('./datasets');

%% load data

heart_table = readtable('./datasets/heart_failure_clinical_records_dataset.csv');
head(heart_table) % preview top 8 rows of data

%% split features and labels

heart_mat = table2array(heart_table);
heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

%% feature normalisation

%% SVM linear kernal

%% SVM Gaussian RBF

%% SVM Polynomial