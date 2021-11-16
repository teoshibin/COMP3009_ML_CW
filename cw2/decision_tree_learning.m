function [best_attribute, best_threshold] = Choose_Attribute(features, targets)
    for attribute = 1:width(features)
        attribute_column = features(:,attribute)
        distinct_value = unique(attribute_column)
        
        for value = distinct_value
            attribute_column
        end
        
    end
    

end

function I = calculateEntropy(positive,negative,threshold)
    p1 = (positive/(positive+negative));
    p2 = (negative/(positive+negative));
    I = -(p1 * log2(p1) + p2 * log2(p2));    
end


function decision_tree = decision_tree_learning(features, labels, task_type =1)
    if task_type == 1:
        
    decision_tree = decision_tree()
    if all(labels == labels(1))
        decision_tree.prediction = labels(1)
    
    else
        [best_attribute, best_threshold] = Choose_Attribute(features, targets);
        left_examples = features(features(:,best_attribute) < best_threshold,:);
        left_targets = targets(features(:,best_attribute) < best_threshold);
        
        decision_tree.kids(end+1) = decion_tree_learning(left_examples, left_targets);
        
        right_examples = features(features(:,best_attribute) >= best_threshold,:);
        right_targets = targets(features(:,best_attribute) >= best_threshold);
        
        decision_tree.kids(end+1) = decion_tree_learning(left_examples, left_targets);
        
       
        
    end


end
