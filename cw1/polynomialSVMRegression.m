%% setup workspace

close all
clear
clc

addpath('./datasets');
addpath('./functions');

rng(1);

%% data loading and normalisation

concrete_table = readtable('./datasets/Concrete_Data.xls');
concrete_mat = table2array(concrete_table);
concrete_mat = shuffleRows(concrete_mat);

concrete_X = concrete_mat(:, 1:end-1);
concrete_Y = concrete_mat(:,end);

concrete_X = minMaxNorm(concrete_X);

norm_concrete_mat = [concrete_X concrete_Y];

%% cross validation

% start stopwatch timer
tic

% variables
k = 10;
inner_k = 5;
hyper_c_boundary = [0.1:0.3:2]; % search space
most_order_boundary = [3:2:7];
epsilon_boundary = [1:2:9];

% records
RMSEs = zeros(1, k);
all_best_hyper_c = zeros(k, inner_k);
all_best_hyper_order = zeros(k, inner_k);
all_best_epsilon = zeros(k, inner_k);

outer_range_indices = partitionIndex(height(norm_concrete_mat),k);

for i = 1:k
    
    % create outer partition
    [outer_test_data, outer_train_data] = splitPartition(norm_concrete_mat, outer_range_indices, i);
    
    % setup inner partition
    inner_range_indices = partitionIndex(height(outer_train_data), inner_k);
    for j = 1:inner_k
       [inner_test_data, inner_train_data] = splitPartition(outer_train_data, inner_range_indices, j);
       
       bestRMSE = Inf;
       
       for c = hyper_c_boundary
           for order = most_order_boundary
               for epsilon = epsilon_boundary
                   model = fitrsvm(inner_train_data(:, 1:end-1), inner_train_data(:,end), ...
                       "BoxConstraint", c, ...
                       "KernelFunction", "polynomial", ...
                       "PolynomialOrder", order, ...
                       "Epsilon", epsilon, ...
                       "Verbose", 0);
                   RMSE = myRMSE(model, inner_test_data(:, 1:end-1), inner_test_data(:, end));
                   
                   if RMSE < bestRMSE
                       bestRMSE = RMSE;
                       all_best_hyper_c(i,j) = c;
                       all_best_hyper_order(i,j) = order;
                       all_best_epsilon(i,j) = epsilon;
                   end
               end
           end
       end
       
        fprintf("\t Inner: %d TestSize: %d TrainSize: %d " + ...
                "RMSE: %f BestC: %f BestPolyOrder: %d BestEpsilon: %d\n", ...
                j, height(inner_test_data), height(inner_train_data), ...
                bestRMSE, all_best_hyper_c(i, j), all_best_hyper_order(i,j), all_best_epsilon(i,j));
    end
    
    most_c = maxCountOccur(all_best_hyper_c(1:i,:));
    most_order = maxCountOccur(all_best_hyper_order(1:i,:));
    most_epsilon = maxCountOccur(all_best_epsilon(1:i,:));
    
    
    model = fitrsvm(outer_test_data(:, 1:end-1), outer_test_data(:, end), ...
        "BoxConstraint",most_c, "KernelFunction", "polynomial", "PolynomialOrder", most_order, "Epsilon", most_epsilon, "Verbose",0);
    
    RMSEs(i) = myRMSE(model, outer_test_data(:, 1:end-1), outer_test_data(:, end));
    
    fprintf("Outer: %d TestSize: %d TrainSize: %d " + ...
            "RMSE: %f MostC: %f MostOrder: %d MostEpsilon: %d\n", ...
            i, height(outer_test_data), height(outer_train_data), ...
            RMSEs(i), most_c, most_order, most_epsilon...
            );
    
end

%% final result

mean_RMSE_polynomial = mean(RMSEs,"all");
best_hyper_constant = maxCountOccur(all_best_hyper_c);
best_hyper_epsilon = maxCountOccur(all_best_epsilon);
best_hyper_order = maxCountOccur(all_best_hyper_order);

fprintf("FinalRMSEPolynomial: %f FinalC: %f FinalOrder: %d FinalEpsilon: %d", ...
      mean_RMSE_polynomial, best_hyper_constant, best_hyper_order, best_hyper_epsilon);

% return elapsed time
toc

%% final model
final_model = fitrsvm(concrete_X, concrete_Y, ...
        "BoxConstraint", best_hyper_constant, ...
        "KernelFunction", "polynomial", ...
        "PolynomialOrder", best_hyper_order, ...
        "Epsilon", best_hyper_epsilon, ...
        "Verbose",1);

%% final model output

number_of_support_vector = numel(find(final_model.IsSupportVector == 1));
support_vector_percentage = number_of_support_vector / numel(final_model.IsSupportVector);
fprintf("supportVectors: %d / %d  svPercentage: %f", number_of_support_vector, height(norm_concrete_mat), support_vector_percentage);
