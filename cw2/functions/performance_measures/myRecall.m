function recall = myRecall(model, test_X, test_Y)

    prediction = model.predict(test_X);

    TP = numel(find(and(prediction, test_Y) == 1));
    FN = numel(find(and(not(prediction), test_Y) == 1));
%     FP = numel(find(and(prediction, not(test_Y)) == 1 ));
%     TN = numel(find(and(not(prediction), not(test_Y)) == 1));

%     precision = TP / (TP + FP);
%     recall = TP / (TP + FN);
    
    recall =  TP / (TP + FN);
end

