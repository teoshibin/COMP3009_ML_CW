function decision_tree = decision_tree_learning(features, targets, task_type)
    %if task_type == 1
       
    if numel(unique(targets)) == 1
        decision_tree.prediction = targets(1);
    
    else
        
        [best_attribute, best_threshold] = Choose_Attribute(features, targets);
        
        left_examples = features(features(:,best_attribute) < best_threshold,:);
        left_targets = targets(features(:,best_attribute) < best_threshold);
        
        decision_tree.kids(end+1) = decision_tree_learning(left_examples, left_targets);
        
        right_examples = features(features(:,best_attribute) >= best_threshold,:);
        right_targets = targets(features(:,best_attribute) >= best_threshold);
        
        decision_tree.kids(end+1) = decision_tree_learning(right_examples, right_targets);
              
    end
end

function [best_attribute, best_threshold] = Choose_Attribute(features, targets)

    % binary classification
    unique_target = unique(targets); 
    positive = sum(targets == 1);
    negetive = sum(targets == 0);
    main_entropy = calculateEntropy(positive, negetive);

    for attribute = 1:width(features)
        
        attribute_column = features(:,attribute);
        distinct_value = unique(attribute_column);
        
        for value = distinct_value
            
            % binary classification
            
            
        end
        
    end
    

end

function I = calculateEntropy(positive, negative)
    p1 = (positive/(positive+negative));
    p2 = (negative/(positive+negative));
    I = -(p1 * log2(p1) + p2 * log2(p2));    
end

function gain = calculateGain(main_entropy, left_entropy, right_entropy, left_proportion, right_proportion)
    gain = main_entropy  - left_proportion * left_entropy - right_proportion * right_entropy;
end
