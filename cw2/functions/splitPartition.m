function [test_data, train_data] = splitPartition(data, range_indices, partition_selection)
    %SPLITPARTITION split actual data using partitionIndex and select one
    %single test partition
    %
    %   Data data before partitioning
    %
    %   Range_indices rows of (1,10) tuples contianing start and end of one
    %   partition
    %
    %   random_perm a mapping for randomizing partition instances
    % 
    %   partition_selection selecting which partition to select
    %   (1=>x=>k) where k is the number of partitions
    %
    
    % single partition [1 2 3] = from range [1 : 3]
    range_list = [range_indices(partition_selection, 1) : range_indices(partition_selection, 2)];
    
    % random number mapping e.g. random_perm = [3 9 2 5 8 7 6 1]
    % single shuffled partition [3 9 2] = random_perm([1 2 3])
    is_test_mat = false(height(data), 1);
    is_test_mat(range_list) = true;

    test_data = data(is_test_mat == 1, :);
    train_data = data(is_test_mat == 0, :);
end

