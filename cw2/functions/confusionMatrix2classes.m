function cm = confusionMatrix2classes(model, test_X, test_Y)

    prediction = model.predict(test_X);

    TP = numel(find(and(prediction, test_Y) == 1));
    FN = numel(find(and(not(prediction), test_Y) == 1));
    FP = numel(find(and(prediction, not(test_Y)) == 1 ));
    TN = numel(find(and(not(prediction), not(test_Y)) == 1));
    
%     fprintf("%d\t%d\n%d\t%d\n", ...
%             TP, FP, FN, TN);
        
    cm = zeros(2,2);
    cm(1,1) = TP;
    cm(1,2) = FP;
    cm(2,1) = FN;
    cm(2,2) = TN;
end