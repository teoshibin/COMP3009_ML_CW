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

%% cross validation

% start stopwatch timer
tic

% changable variables
k = 10;
inner_k = 5;
hyper_c_boundary = [0.1:0.1:2]; % search space
most_order_boundary = [3:2:7];

% fixed variables
accuracies = zeros(1, k);
f1Scores = zeros(1, k);
all_best_hyper_c = zeros(k, inner_k); % each row contain every inner cv best c
all_best_hyper_order = zeros(k, inner_k);

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
        bestF1Score = 0;
                
        % exhaustive search
        % change nesting of nested loops depending on numbers of hyper-parameters
        for c = hyper_c_boundary
            for q = most_order_boundary
                model = fitcsvm( ...
                    inner_train_data(:, 1:end-1), ...
                    inner_train_data(:, end), ...
                    "KernelFunction", "polynomial", ...
                    "BoxConstraint", c,  ...
                    "PolynomialOrder", q ...
                    );
        
                f1Score = myFOneScore(model, inner_test_data(:, 1:end-1), inner_test_data(:, end));
                inner_accuracy = myAccuracy(model, inner_test_data(:, 1:end-1), inner_test_data(:, end));
            
                % parameter tuning using f1Score
                if f1Score > bestF1Score
                    all_best_hyper_c(i, j) = c;
                    all_best_hyper_order(i, j) = q;
                    bestF1Score = f1Score;
                end
            end
        end
        
        fprintf("\t Inner: %d TestSize: %d TrainSize: %d " + ...
                "Accuracy: %f F1Score: %f BestC: %f BestQ: %d\n", ...
                j, height(inner_test_data), height(inner_train_data), ...
                inner_accuracy, bestF1Score, all_best_hyper_c(i, j), all_best_hyper_order(i,j));
    
    end
   
    % get tuned c & sigma
    most_c = maxCountOccur(all_best_hyper_c(1:i,:));
    most_order = maxCountOccur(all_best_hyper_order(1:i,:));
    
    % train using tuned c & sigma
    model = fitcsvm( ...
        outer_train_data(:, 1:end-1), ...
        outer_train_data(:, end), ...
        "KernelFunction", "polynomial", ...
        "BoxConstraint", most_c, ...
        "PolynomialOrder", most_order, ...
        "Verbose", 0 ...
        );
    
    % calculate accuracy
    f1Scores(i) = myFOneScore(model, outer_test_data(:, 1:end-1), outer_test_data(:,end));
    accuracies(i) = myAccuracy(model, outer_test_data(:, 1:end-1), outer_test_data(:,end));
    
    fprintf("Outer: %d TestSize: %d TrainSize: %d " + ...
            "Accuracy: %f F1Score: %f MostC: %f MostOrder: %d\n", ...
            i, height(outer_test_data), height(outer_train_data), ...
            accuracies(i), f1Scores(i), most_c, most_order ...
            );
    
end

%% final result

mean_accuracy = mean(accuracies, "all");
mean_f1 = mean(f1Scores, "all");
best_hyper_constant = maxCountOccur(all_best_hyper_c);
best_hyper_order = maxCountOccur(all_best_hyper_order);

fprintf("FinalAccuracy: %f FinalF1: %f FinalC: %d FinalOrder: %d ", ...
    mean_accuracy, mean_f1, best_hyper_constant, best_hyper_order);

% return elapsed time
toc

%% final model

final_model = fitcsvm( ...
    heart_X, ...
    heart_Y, ...
    "KernelFunction", "polynomial", ...
    "BoxConstraint", best_hyper_constant, ...
    "PolynomialOrder", best_hyper_order, ...
    "Verbose", 1 ...
    );

%% final model output

number_of_support_vector = numel(find(final_model.IsSupportVector == 1));
support_vector_percentage = number_of_support_vector / numel(final_model.IsSupportVector);
fprintf("supportVectors: %d svPercentage: %f", number_of_support_vector, support_vector_percentage);
