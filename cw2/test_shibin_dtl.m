%% classification
close all
clear
clc

rng(1);

addpath('./datasets');
addpath(genpath('./functions'));

heart_table = readtable('heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);
heart_mat = shuffleRows(heart_mat);

heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

fprintf("Total Instances: %d\n", height(heart_X));
tree = shibin_dtl(heart_X, heart_Y, "Classification", heart_table.Properties.VariableNames);

DrawDecisionTree(tree);

answer = predict(tree, heart_X);
accuracy = myAccuracy(heart_Y, answer);
fprintf("Accuracy: %.2f%%\n", accuracy*100);


%% regression
close all
clear
clc

rng(1);

addpath('./datasets');
addpath(genpath('./functions'));

concrete_table = readtable('Concrete_Data.xls');
concrete_mat = table2array(concrete_table);
concrete_mat = shuffleRows(concrete_mat);

concrete_X = concrete_mat(:, 1:end-1);
concrete_Y = concrete_mat(:,end);

fprintf("Total Instances: %d\n", height(concrete_X));
tree = shibin_dtl(concrete_X, concrete_Y, "Regression", concrete_table.Properties.VariableNames);

DrawDecisionTree(tree);

answer = predict(tree, concrete_X);
accuracy = myAccuracy(concrete_Y, answer);
fprintf("Accuracy: %.2f%%\n", accuracy*100);

