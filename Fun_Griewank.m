function f = Fun_Griewank(x)
% the Grtiewank function
% xi = [-600,600]
d = size(x,2);
f = sum(x.^2/4000,2) - prod(cos(x./sqrt(1:d)),2) + 1;
end
