function out = maxCountOccur(mat)
    A = reshape(mat, [], 1); % reshape anything into veritical column
    [frequency,number] = groupcounts(A);
    index = frequency==max(frequency);
    out = number(index);
    out = out(1);
end

