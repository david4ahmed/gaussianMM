classdef plotGMM
    methods (Static)
        function ret = show_ellipsoid(means, covariances, npts)
            [v,d]=eig(covariances); 
            d = sqrt(d);
            [x,y,z] = sphere(npts);
            s = transpose([x(:) y(:) z(:)]);
            data = v*d*s;
            t = repmat(means, 1, size(s,2));
            data = data + t; 
            % x = data(1,:);
            % y = data(2,:);
            % z = data(3,:);
            ret = surf(gca, x, y, z);
        end
    end
end