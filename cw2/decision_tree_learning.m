function decision_tree = decision_tree_learning(features, targets, task_type)
    decision_tree = decision_tree() 
    % Classification Tree
    if task_type == 1
        if numel(unique(targets)) == 1
            decision_tree.prediction = targets(1);

        else

            [best_attribute, best_threshold] = Choose_Attribute(features, targets);
            decision_tree.op = best_attribute

            left_examples = features(features(:,best_attribute) < best_threshold,:);
            left_targets = targets(features(:,best_attribute) < best_threshold);

            decision_tree.kids(end+1) = decision_tree_learning(left_examples, left_targets);

            right_examples = features(features(:,best_attribute) >= best_threshold,:);
            right_targets = targets(features(:,best_attribute) >= best_threshold);

            decision_tree.kids(end+1) = decision_tree_learning(right_examples, right_targets);
        end
        
    end
    % Regression Tree
    if task_type == 0
        
            
    end
end

function [best_attribute, best_threshold] = Choose_Attribute(features, targets, task_type)
    if task_tyope == 1:
        % binary classification
        unique_target = unique(targets); 
        positive = sum(targets == 1);
        negetive = sum(targets == 0);
        main_entropy = calculateEntropy(positive, negetive);

        for attribute = 1:width(features)

            attribute_column = features(:,attribute);
            distinct_value = unique(attribute_column);
            if distinct_value == 2
                for value = distinct_value
                    % binary classification
                end

            end
        end
    

end

function I = calculateEntropy(positive, negative)
    p1 = (positive/(positive+negative));
    p2 = (negative/(positive+negative));
    I = -(p1 * log2(p1) + p2 * log2(p2));    
end

function R = calculateRemainder(attribute, targets)
    total = 0;
    for unique_value = unique(atrribute)
        weight = height(targets(attribute == unique_value))/height(attribute);
        positives = sum(targets(attribute == unique_value));
        negatives = height(targets(attribute == unique_value)) - positives;
        total = total + weight*calculateEntropy(positives, negatives);
    end
    R = total;
end


function gain = calculateGain(attribute, targets)
    gain = calculateEntrophy(height(targets(targets == 1)),height(targets(targets == 0))) - Remainder(attribute, targets);
end
