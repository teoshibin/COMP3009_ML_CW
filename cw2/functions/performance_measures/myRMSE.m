function RMSE = myRMSE(model, test_X, test_Y)
    
    prediction = model.predict(test_X);
    RMSE = sqrt(mean((prediction - test_Y).^2));

end