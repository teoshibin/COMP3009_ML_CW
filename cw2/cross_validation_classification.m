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
training_data = [heart_X heart_Y];

k = 10;
inner_k = 5;
hyper_depth = [2:10];

%result store
accuracies = zeros(1,k);
f1Scores = zeros(1,k);
precisions = zeros(1,k);
recall = zeros(1,k);
all_best_hyper_depth = zeros(k, inner_k);

%setup partition
outer_range_indices = partitionIndex(height(training_data), k);

for i = 1:k
    %create outer partition
    [outer_test_data, outer_train_data] = splitPartition(training_data, outer_range_indices, i);
    
    %setup inner partition
    inner_range_indices = partitionIndex(height(outer_train_data), inner_k);

    for j = 1:inner_k

        %create inner partition
        [inner_test_data, inner_train_data] = splitPartition(outer_train_data, inner_range_indices,j);

        bestF1Score = 0;

        for d = hyper_depth
            tree = shibin_dtl(inner_train_data(:,1:end-1), inner_train_data(:,end),"Classification", ...
                heart_table.Properties.VariableNames, d);

            predicted = predict(tree, inner_test_data(:, 1:end-1));

            inner_accuracy = myAccuracy(inner_test_data(:, end), predicted);

            [f1Score, precision, recall] = myFOneScore(inner_test_data(:,end),predicted);

            if f1Score > bestF1Score
                all_best_hyper_depth(i,j) = d;
                bestF1Score = f1Score;
            end   

        end

        fprintf("\t Inner: %d TestSize: %d TrainSize: %d Accuracy: %f F1Score: %f BestD: %d\n"  ...
            ,j,height(inner_test_data),height(inner_train_data) ...
            ,inner_accuracy,f1Score, all_best_hyper_depth(i,j));
    end
    
    %tuned depth
    most_d = maxCountOccur(all_best_hyper_depth(1:i,:));

    %train using tuned depth
    tree = shibin_dtl(outer_train_data(:,1:end-1), outer_train_data(:,end),"Classification" ...
        , heart_table.Properties.VariableNames, most_d);

    predicted = predict(tree, outer_test_data(:, 1:end-1));

    %storing result
    accuracies(i) = myAccuracy(outer_test_data(:, end), predicted);

    [f1Scores(i), precisions(i), recall(i)] = myFOneScore(outer_test_data(:,end),predicted);

    fprintf("Outer: %d TestSize: %d TrainSize: %d Accuracy: %f F1Score: %f Precision: %f Recall: %f " + ...
        "MostDepth: %d\n",i, height(outer_test_data), height(outer_train_data), accuracies(i), ...
        f1Scores(i), precisions(i), recall(i), most_d);

end

% Final result
best_hyper_depth = maxCountOccur(all_best_hyper_depth);
mean_f1 = mean(f1Scores, "all");
mean_accuracy = mean(accuracies, "all");

fprintf("FinalAccuracy: %f FinalF1: %f Finald: %d\n", mean_accuracy, mean_f1, best_hyper_depth);
