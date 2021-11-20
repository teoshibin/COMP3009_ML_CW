%setup workspace
close all
clear
clc

addpath('./datasets');
addpath(genpath('./functions'));

rng(1);

%data loading
concrete_table = readtable('Concrete_Data.xls');
concrete_mat = table2array(concrete_table);
concrete_mat = shuffleRows(concrete_mat);

concrete_X = concrete_mat(:, 1:end-1);
concrete_Y = concrete_mat(:,end);

training_data = [concrete_X concrete_Y];

k = 10;
inner_k = 5;
hyper_depth = [2:10];

%result store
RMSEs = zeros(1,k);
all_best_hyper_depth = zeros(k,inner_k);

%setup partition
outer_range_indices = partitionIndex(height(training_data),k);

for i = 1:k
    %create outer partition
    [outer_test_data, outer_train_data] = splitPartition(training_data, outer_range_indices, i);

    %setup inner partition
    inner_range_indices = partitionIndex(height(outer_train_data), inner_k);

    for j = 1:inner_k

        %create inner partition
        [inner_test_data, inner_train_data] = splitPartition(outer_train_data, inner_range_indices, j);

        bestRMSE = Inf;

        for d = hyper_depth
            tree = shibin_dtl(inner_train_data(:,1:end-1), inner_train_data(:,end),"Regression", ...
                concrete_table.Properties.VariableNames, d);

            predicted = predict(tree, inner_test_data(:, 1:end-1));

            RMSE = myRMSE(inner_test_data(:, end), predicted);

            if RMSE < bestRMSE
                all_best_hyper_depth(i,j) = d;
                bestRMSE = RMSE;
            end
        end
        fprintf("\t Inner: %d TestSize: %d TrainSize: %d RMSE: %f BestD: %d\n"  ...
            ,j,height(inner_test_data),height(inner_train_data) ...
            ,bestRMSE, all_best_hyper_depth(i,j));
    end

    %tuned depth
    most_d = maxCountOccur(all_best_hyper_depth(1:i,:));

    %train using tuned depth
    tree = shibin_dtl(outer_train_data(:,1:end-1), outer_train_data(:,end),"Regression", ...
                concrete_table.Properties.VariableNames, most_d);

    predicted = predict(tree, outer_test_data(:, 1:end-1));

    %storing result
    RMSEs(i) = myRMSE(outer_test_data(:, end), predicted);

    fprintf("Outer: %d TestSize: %d TrainSize: %d RMSE: %f" + ...
        " MostDepth: %d\n",i, height(outer_test_data), height(outer_train_data), RMSEs(i), most_d);

end

%Final result
best_hyper_depth = maxCountOccur(all_best_hyper_depth);
mean_RMSE = mean(RMSEs, "all");

fprintf("FinalRMSE: %f FinalD: %d\n",mean_RMSE, best_hyper_depth);