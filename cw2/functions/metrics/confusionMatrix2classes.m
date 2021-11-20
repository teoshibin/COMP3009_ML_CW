function cm = confusionMatrix2classes(actual_Y, predicted_Y)

    TP = numel(find(and(predicted_Y, actual_Y) == 1));
    FN = numel(find(and(not(predicted_Y), actual_Y) == 1));
    FP = numel(find(and(predicted_Y, not(actual_Y)) == 1 ));
    TN = numel(find(and(not(predicted_Y), not(actual_Y)) == 1));
    
%     fprintf("%d\t%d\n%d\t%d\n", ...
%             TP, FP, FN, TN);
        
    cm = zeros(2,2);
    cm(1,1) = TP;
    cm(1,2) = FP;
    cm(2,1) = FN;
    cm(2,2) = TN;
    
end