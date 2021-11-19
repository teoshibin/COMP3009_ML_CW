%% test run

close all
clear
clc

addpath('./datasets');
addpath('./functions');

heart_table = readtable('heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);
heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

decision_tree = decision_tree_learning(heart_X, heart_Y, 1, heart_table);

DrawDecisionTree(decision_tree);

answer = predict(decision_tree, heart_X);
accuracy = myAccuracy(heart_Y, answer);
fprintf("Accuracy: %.2f%%\n", accuracy*100);

%% main function
function tree = decision_tree_learning(features, targets, task_type, table)
    tree = decision_tree() ;
    % Classification Tree
    if task_type == 1
        if numel(unique(targets)) == 1
            tree.prediction = targets(1);
        else
            [best_attribute, best_threshold] = Choose_Attribute(features, targets, task_type);
            if best_attribute == 0
                tree.prediction = mode(targets);
                return
            end
            
            tree.op = table.Properties.VariableNames{best_attribute};
            tree.threshold = best_threshold;
            tree.attribute = best_attribute;

            left_examples = features(features(:,best_attribute) < best_threshold,:);
            left_targets = targets(features(:,best_attribute) < best_threshold);

            tree.kids{1} = decision_tree_learning(left_examples, left_targets, task_type, table);

            right_examples = features(features(:,best_attribute) >= best_threshold,:);
            right_targets = targets(features(:,best_attribute) >= best_threshold);

            tree.kids{2} = decision_tree_learning(right_examples, right_targets, task_type, table);
        end
        
    end
    % Regression Tree
    if task_type == 0
        
            
    end
end

function [best_attribute, best_threshold] = Choose_Attribute(features, targets, task_type)
    % Classification Tree
    if task_type == 1 
        
        max_gain = 0;
        best_attribute = 0;
        best_threshold = -1;
        
        for attribute = 1:width(features) % Lopp through all the attribute
            attribute_column = features(:,attribute);
            attribute_values = unique(attribute_column); 
            if numel(attribute_values) == 1
                continue
            end
            thresholds = (attribute_values(1:end-1) + attribute_values(2:end)) / 2;
            for threshold = transpose(thresholds)
                gain = calculateGain(attribute_column, targets, threshold);
                if gain > max_gain
                    max_gain = gain;
                    best_attribute = attribute;
                    best_threshold = threshold;
                end
            end
        end
    end
    % Regression Tree
    if task_type == 0
        
        
    end
end

function I = calculateEntropy(positive, negative)
    p1 = (positive/(positive+negative));
    p2 = (negative/(positive+negative));
    I = -(p1 * log2(p1) + p2 * log2(p2));    
end

function R = calculateRemainder(attribute, targets, threshold)
    total = 0;
    
    % The gain for left kid.
    weight = height(targets(attribute < threshold))/height(attribute);
    positives = sum(targets(attribute < threshold));
    negatives = height(targets(attribute < threshold)) - positives;
    total = total + weight*calculateEntropy(positives, negatives);
    
    % The gain for right kid.
    weight = height(targets(attribute >= threshold))/height(attribute);
    positives = sum(targets(attribute >= threshold));
    negatives = height(targets(attribute >= threshold)) - positives;
    total = total + weight*calculateEntropy(positives, negatives);

    R = total;
end

function gain = calculateGain(attribute, targets, threshold)
    gain = calculateEntropy( sum(targets == 1), sum(targets == 0) ) - calculateRemainder(attribute, targets, threshold);
end

function outputs = predict(tree, inputs)
    outputs = []
    root = tree
    for i = 1:height(inputs)
        tree = root
        input = inputs(i,:);
        while isempty(tree.prediction)
            tree.attribute;
            if input(tree.attribute) < tree.threshold
                tree = tree.kids{1};
                disp("left")
            elseif input(tree.attribute) >= tree.threshold
                tree = tree.kids{2};
                disp("right")
            end
        end
        output = tree.prediction;
        outputs = cat(1,outputs,output);
    end
    
end


