function accuracy = myAccuracy(model, test_X, test_Y)
    prediction = model.predict(test_X);
    correct = numel(find(prediction == test_Y));
    all = height(test_X);
    accuracy =  correct / all;
end

