function out = shibin_dtl(features, targets, task_type, feature_names)

    arguments
        features (:,:) {mustBeNumeric}
        targets (:,1) {mustBeNumeric}
        task_type {mustBeMember(task_type,["Classification","Regression"])}
        feature_names (1,:) string
    end

    tree = struct_decision_tree();
        
        if numel(unique(targets)) < 2 
            
            % only one type of label left then assign prediction label
            tree.prediction = targets(1);
           % fprintf("Completed Instances: %d\n", numel(targets));
            
        else
            % find best threshold and attribute
            [best_attribute, best_threshold] = Choose_Attribute(features, targets, task_type);
            tree.op = feature_names(best_attribute);
            tree.threshold = best_threshold;
            tree.attribute = best_attribute;

            fprintf("attribute: %d threshold %.4f\n", best_attribute, best_threshold);
            
            if best_threshold == 0
                disp("hi");
            end
            
            % if left examples turn into empty matrix this recursion wont
            % be able to stop
            % split left
            left_examples = features(features(:,best_attribute) < best_threshold,:);
            left_targets = targets(features(:,best_attribute) < best_threshold);
            
            % calculate left child node
            left_node = shibin_dtl(left_examples, left_targets, task_type,feature_names);
            
            % split right
            right_examples = features(features(:,best_attribute) >= best_threshold,:);
            right_targets = targets(features(:,best_attribute) >= best_threshold);

            % calculate right child node
            right_node = shibin_dtl(right_examples, right_targets, task_type,feature_names);
                      
            tree.kids = {left_node right_node};
            
        end
        
        out = tree;
end

function [best_attribute, best_threshold] = Choose_Attribute(features, targets, task_type)
       
    if task_type == "Classification"
        
        % calculate root entropy
        root_positive = sum(targets == 1);
        root_negetive = sum(targets == 0);
        root_entropy = calculateEntropy(root_positive, root_negetive);

        % store classification results
        gains = zeros(1, width(features));
        thresholds = zeros(1, width(features));

        % for all attribute
        for attribute_index = 1:width(features)

            attribute_column = features(:, attribute_index);
            distinct_value = unique(attribute_column);

            % distinct values cannot be less than 2 otherwise cant split
            if numel(distinct_value) ~= 1

                threshold_list = zeros(1, numel(distinct_value) - 1);

                % find the mean between each unique value
                for i = 1:width(threshold_list)
                    threshold_list(1, i) = (distinct_value(i) + distinct_value(i+1)) / 2;
                end

                % for all unique value in this attribute
                for threshold = threshold_list

                    % calcuate information gain
                    remainder = entropyRemainder(features, targets, attribute_index, threshold);
                    gain = root_entropy - remainder;

                    if gain > gains(attribute_index)
                        gains(attribute_index) = gain;
                        thresholds(attribute_index) = threshold;
                    end
                end
            end
        end

        % find best attribute
        best_attribute = find(gains == max(gains), 1);
        best_threshold = thresholds(best_attribute);

    elseif task_type == "Regression"

        % calculate root variance
        root_variance = safeVar(targets);

        % store classification results
        reductions = inf(1, width(features));
        thresholds = zeros(1, width(features));

        % for all attribute
        for attribute_index = 1:width(features)

            attribute_column = features(:, attribute_index);
            distinct_value = unique(attribute_column);

            % distinct values cannot be less than 2 otherwise cant split
            if numel(distinct_value) ~= 1

                threshold_list = zeros(1, numel(distinct_value) - 1);

                % find the mean between each unique value
                for i = 1:width(threshold_list)
                    threshold_list(1, i) = (distinct_value(i) + distinct_value(i+1)) / 2;
                end

                % for all unique value in this attribute
                for threshold = threshold_list

                    % calcuate information gain
                    remainder = varianceRemainder(features, targets, attribute_index, threshold);
                    reduction = root_variance - remainder;

                    if reduction < reductions(attribute_index)
                        reductions(attribute_index) = reduction;
                        thresholds(attribute_index) = threshold;
                    end
                end
            end
        end

        % find best attribute
        best_attribute = find(reductions == min(reductions), 1);
        best_threshold = thresholds(best_attribute);
        
    end

end

function I = calculateEntropy(positive, negative)
    p1 = (positive/(positive+negative));
    p2 = (negative/(positive+negative));
    I = -(p1 * safeLog2(p1) + p2 * safeLog2(p2));    
end

function out = safeLog2(value)
    if value == 0
        out = 0;
    else
    	out = log2(value);
    end
end

function out = safeVar(value)
    result = var(value, 1);
    if isnan(result)
        out = 0;
    else
        out = result;
    end
end

function out = entropyRemainder(features, targets, selected_feature, threshold)

    % splited labels
    left_child = targets(features(:, selected_feature) < threshold);
    right_child = targets(features(:, selected_feature) >= threshold);

    % left child
    left_positive = numel(left_child(left_child == 1));
    left_negative = numel(left_child(left_child == 0));
    left_entropy = calculateEntropy(left_positive, left_negative);

    % right_child
    right_positive = numel(right_child(right_child == 1));
    right_negative = numel(right_child(right_child == 0));
    right_entropy = calculateEntropy(right_positive, right_negative);

    % calculate weights
    left_weight = numel(left_child) / numel(targets);
    right_weight = numel(right_child) / numel(targets);

    % calcuate information gain
    out = left_weight * left_entropy + right_weight * right_entropy;
end

function out = varianceRemainder(features, targets, selected_feature, threshold)
    
    % splited labels
    left_child = targets(features(:, selected_feature) < threshold);
    right_child = targets(features(:, selected_feature) >= threshold);

    % children variance
    left_variance = safeVar(left_child);
    right_variance = safeVar(right_child);

    % calculate weights
    left_weight = numel(left_child) / numel(targets);
    right_weight = numel(right_child) / numel(targets);

    % calcuate information gain
    out = left_weight * left_variance + right_weight * right_variance;
    
end
