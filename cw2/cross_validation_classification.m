close all
clear
clc

addpath('./datasets');
addpath(genpath('./functions'));

rng(1);

heart_table = readtable('./datasets/heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);
heart_mat = shuffleRows(heart_mat);

heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

k = 10;

training_data = [heart_X heart_Y];
outer_range_indices = partitionIndex(height(training_data), k);

for i = 1:k
    [outer_test_data, outer_train_data] = splitPartition(training_data, outer_range_indices, i);
    tree = shibin_dtl(outer_train_data(:,1:end-1), outer_train_data(:,end),"Classification", heart_table.Properties.VariableNames);
    answer = predict(tree, outer_test_data(:, 1:end-1));
    accuracy = myAccuracy(outer_test_data(:, end), answer);
    %[testF1, test1,test2] = f1Score(tree,outer_test_data(:,1:end-1),outer_test_data(:,end));
    [f1Score, precision, recall] = myFOneScore(outer_test_data(:,end),answer);
    fprintf("Accuracy:%f F1Score:%f Precision:%f Recall:%f\n",accuracy,f1Score,precision,recall);
end

