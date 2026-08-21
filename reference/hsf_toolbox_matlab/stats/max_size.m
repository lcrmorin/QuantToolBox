function [r,c,e] = max_size(x1,x2,x3,x4,x5)

    [r1,c1] = size(x1);
    [r2,c2] = size(x2);
    r = max(r1,r2);
    c = max(c1,c2);

    if nargin > 2
        [r3,c3] = size(x3);
        r = max(r,r3);
        c = max(c,c3);
    end

    if nargin > 2
        [r3,c3] = size(x3);
        r = max(r,r3);
        c = max(c,c3);
    end

    if nargin > 3
        [r4,c4] = size(x4);
        r = max(r,r4);
        c = max(c,c4);
    end

    if nargin > 4
        [r5,c5] = size(x5);
        r = max(r,r5);
        c = max(c,c5);
    end

    if nargout == 3
        e = ones(r,c);
    end

end
