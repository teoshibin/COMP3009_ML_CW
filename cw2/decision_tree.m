function tree = decision_tree(op,kids,prediction,attribute,threshold)
    tree = struct('op',op,'kids',kids,'prediction',prediction,'attribute',attribute,'threshold',threshold);
end