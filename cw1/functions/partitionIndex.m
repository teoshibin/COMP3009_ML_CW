function out = partitionIndex(instance_size, k)
% Input number of instances and the number of bins to split
%   instance_size = scalar value, dataset number of instances
%   k = scalar value, number of partition

    partitionIndices = zeros(k, 2);
    itemsNumPerFold = round(instance_size / k); 

    for i = 1:k
        partitionIndices(i, :) = [(i-1)*itemsNumPerFold+1 i*itemsNumPerFold];
    end
    partitionIndices(k, 2) = instance_size;
    
    out = partitionIndices;
end

