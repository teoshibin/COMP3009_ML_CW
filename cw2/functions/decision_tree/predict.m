% predict function for decision tree
%   tree    decision tree struct defined in empty_decision_tree
%   inputs  NxM predictors N: instances M: attributes

function outputs = predict(tree, inputs)

    outputs = [];
    root = tree;
    
    % for each instances
    for i = 1:height(inputs)
        
        tree = root;            % copy of root
        input = inputs(i,:);    % one instance
        
        % do until leaf node
        while isempty(tree.prediction)
            tree.attribute;
            if input(tree.attribute) < tree.threshold
                tree = tree.kids{1};
            elseif input(tree.attribute) >= tree.threshold
                tree = tree.kids{2};
            end
        end
        
        output = tree.prediction;
        outputs = cat(1,outputs,output);
        
    end
end