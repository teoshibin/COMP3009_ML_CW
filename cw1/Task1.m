%% Linear SVM for Classification
% Set up the enviroment.
close all
clear
clc

addpath('./datasets');
addpath('./functions');

rng(1);

% Data Loading and Pre-processing.
heart_table = readtable('./datasets/heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);
heart_mat = shuffleRows(heart_mat);

heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

heart_X = minMaxNorm(heart_X);

norm_heart_mat = [heart_X heart_Y];

% Train the model.
model = fitcsvm(heart_X, heart_Y,...
                "KernelFunction", "linear", ...
                'BoxConstraint', 1);

% Evaluate the results.
accuracy = myAccuracy(model,heart_X,heart_Y);
hist(heart_Y);
cm = confusionMatrix2classes(model, heart_X, heart_Y);
% cm = confusionmat(heart_Y, model.predict(heart_X));
confusionchart(cm);
fprintf("Accuracy: %f", accuracy);

%% Linear SVM for Regression
% Set up the enviroment.
close all
clear
clc

addpath('./datasets');
addpath('./functions');

rng(1);

% Data Loading and Pre-processing.
concrete_table = readtable('./datasets/Concrete_Data.xls');
concrete_mat = table2array(concrete_table);
concrete_mat = shuffleRows(concrete_mat);

concrete_X = concrete_mat(:, 1:end-1);
concrete_Y = concrete_mat(:,end);

concrete_X = minMaxNorm(concrete_X);


% Train the model.
Epsilons = [0:20];
line = [1:100];
RMSEs = zeros(21);
i = 1
for epsilon = Epsilons
    model = fitrsvm(concrete_X, concrete_Y, ...
                    "KernelFunction", "linear", ...
                    "Epsilon", epsilon,...
                    "Verbose", 1);
    figure;
    plot(model.predict(concrete_X),concrete_Y,'o',line,line);
    RMSE = myRMSE(model, concrete_X, concrete_Y);
    title("RMSE:", RMSE);
    RMSEs(epsilon+1) = RMSE;
    i = i+1;
end

% print the results.
for i = Epsilons
    fprintf("Epsilon: %d RMSE: %d\n", ...
            i, RMSEs(i+1));
end