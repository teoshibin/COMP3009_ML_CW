%% setup workspace

close all
clear
clc

addpath('./datasets');
addpath('./functions');

rng(1);

%% data loading and normalisation

heart_table = readtable('./datasets/heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);
heart_mat = shuffleRows(heart_mat);

heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

heart_X = minMaxNorm(heart_X);

norm_heart_mat = [heart_X heart_Y];

%% training

% changable variables
k = 10;
inner_k = 5;
hyper_c_boundary = [1:100]; % search space

% fixed variables
accuracies = zeros(1, k);
all_best_hyper_c = zeros(k, inner_k); % each row contain every inner cv best c


% setup outer partition
outer_range_indices = partitionIndex(height(norm_heart_mat), k); % each rows contain [partition_start partition_ends]

for i = 1:k
    
    % create outer partition       
    [outer_test_data, outer_train_data] = splitPartition(norm_heart_mat, outer_range_indices, i);
    
    % setup inner partition
    inner_range_indices = partitionIndex(height(outer_train_data), inner_k);
    
    for j = 1:inner_k
        
        % create inner partition
        [inner_test_data, inner_train_data] = splitPartition(outer_train_data, inner_range_indices, j);

        % variables
        bestTuningAccuracy = 0;
                
        % exhaustive search
        % change nesting of nested loops depending on numbers of hyper-parameters
        for c = hyper_c_boundary
                    
            model = fitcsvm(inner_train_data(:, 1:end-1), inner_train_data(:, end), ...
                'BoxConstraint', c, "KernelFunction", "linear", "Verbose", 0);
        
            prediction = model.predict(inner_test_data(:, 1:end-1));
            correct = numel(find(prediction == inner_test_data(:,end)));
            all = height(inner_test_data);
            tuningAccuracy =  correct / all;
            
            % parameter tuning using accuracy
            if tuningAccuracy > bestTuningAccuracy
                all_best_hyper_c(i, j) = c;
                bestTuningAccuracy = tuningAccuracy;
            end
            
        end
                
        % save accuracy
        accuracies(i) = accuracies(i) + (bestTuningAccuracy / inner_k);
        
        fprintf("\t Inner: %d TestSize: %d TrainSize: %d Accuracy: %f BestC: %d\n", ...
            j, height(inner_test_data), height(inner_train_data), bestTuningAccuracy, all_best_hyper_c(i, j));
    
    end
   
    fprintf("Outer: %d TestSize: %d TrainSize: %d Accuracy: %f\n", ...
        i, height(outer_test_data), height(outer_train_data), accuracies(i));
    
end
%%
best_hyper_constant = round(mean(all_best_hyper_c, "all"));
mean_accuracy = mean(accuracies);

fprintf("Accuracy: %f BestC: %d\n", mean_accuracy, best_hyper_constant);

%% final model

final_model = fitcsvm(heart_X, heart_Y, 'BoxConstraint', best_hyper_constant, "KernelFunction", "linear");

number_of_support_vector = numel(find(final_model.IsSupportVector == 1));
support_vector_percentage = number_of_support_vector / numel(final_model.IsSupportVector);

fprintf("supportVectors: %d svPercentage: %f\n", number_of_support_vector, support_vector_percentage);





