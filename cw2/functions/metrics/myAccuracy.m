function accuracy = myAccuracy(actual_Y, predict_Y)
    correct = numel(find(predict_Y == actual_Y));
    all = height(actual_Y);
    accuracy =  correct / all;
end

