function [best_attribute, best_threshold] = Choose_Attribute(features, targets)
    

end

function decision_tree = decision_tree_learning(features, labels)
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
