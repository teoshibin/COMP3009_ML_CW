
close all
clear
clc

addpath('./datasets');
addpath(genpath('./functions'));

heart_table = readtable('heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);
heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

fprintf("Total Instances: %d\n", height(heart_X));
tree = shibin_dtl(heart_X, heart_Y, "Classification", heart_table.Properties.VariableNames);

DrawDecisionTree(tree);

answer = predict(tree, heart_X);
accuracy = myAccuracy(heart_Y, answer);
fprintf("Accuracy: %.2f%%\n", accuracy*100);
