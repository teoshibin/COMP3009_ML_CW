
close all
clear
clc

addpath('./datasets');
addpath(genpath('./functions'));

heart_table = readtable('heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);
heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

decision_tree = decision_tree_learning(heart_X, heart_Y, 1, heart_table);

DrawDecisionTree(decision_tree);

answer = predict(decision_tree, heart_X);
accuracy = myAccuracy(heart_Y, answer);
fprintf("Accuracy: %.2f%%\n", accuracy*100);
