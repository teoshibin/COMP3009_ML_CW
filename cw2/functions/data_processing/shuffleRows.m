function out = shuffleRows(data)
    out = data(randperm(size(data, 1)), :);
end

