function out = partitionIndex(instance_size, k)
% Input number of instances and the number of bins to split
%   instance_size = scalar value, dataset number of instances
%   k = scalar value, number of partition
%   out = [P1_start P1_end;
%          P2_start P2_end;
%          ...]
%   each row contians the start and ends index of a single partition
%   the last partition may contain few less or few more intances depending
%   on the number of instances

    partitionIndices = zeros(k, 2);
    itemsPerFold = round(instance_size / k); 

    for i = 1:k
        partitionIndices(i, :) = [(i-1)*itemsPerFold+1 i*itemsPerFold];
    end
    partitionIndices(k, 2) = instance_size;
    
    out = partitionIndices;
end

