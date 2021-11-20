function [F1_score, precision, recall] = myFOneScore(actual_Y, predicted_Y)

    %prediction = model.predict(test_X);

    TP = numel(find(and(predicted_Y, actual_Y) == 1));
    FN = numel(find(and(not(predicted_Y), actual_Y) == 1));
    FP = numel(find(and(predicted_Y, not(actual_Y)) == 1 ));
    TN = numel(find(and(not(predicted_Y), not(actual_Y)) == 1));

    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    if (isnan(recall))
        recall = 0;
    end
    
    F1_score =  2*TP / (2*TP + FP + FN);
end

