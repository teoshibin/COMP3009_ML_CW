function out = getAlph()
    a = ['A':'Z'];
    a = string(a);
    a = split(a,"");
    a = a(2:end-1);
    out = a;
end

