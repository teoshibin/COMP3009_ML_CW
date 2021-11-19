
heart_table = readtable('./datasets/heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);

heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

fprintf("Total Instances: %d\n", height(heart_X));
tree = shibin_dtl(heart_X, heart_Y, "Classification");

DrawDecisionTree(tree);