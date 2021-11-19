function out = minMaxNorm(X)
%MINMAXNORM min max normalisation with 5 95 percentile
    
    norm = zeros(size(X));
    for i = 1:width(X)
        min5 = prctile(X(:,i), 5);           % use 5% and 95% percentile as the min and max values.
        max95 = prctile(X(:,i), 95);
        norm(:,i) = (X(:,i) - min5)/(max95 - min5);   % min-max normalisation
        norm(:,i) = min(norm(:,i), 1);         % clip the values higher than 1 and lower than 0. 
        norm(:,i) = max(norm(:,i), 0);
    end
    
    out = norm;
end

