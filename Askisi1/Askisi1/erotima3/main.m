randn('seed',0) %initialization of the random number generator
m1 = [0 0 0]';
m2 = [0.5 0.5 0.5]';

P1 = 0.5;
P2 = 0.5;

m = [m1 m2];

S = [0.8 0.01 0.01;0.01 0.2 0.01;0.01 0.01 0.2];

N = 10000;

X = [mvnrnd(m1',S,N/2); mvnrnd(m2',S,N/2)]'; %3XN matrix

t=[ones(1,N/2) 2*ones(1,N/2)];


out_eucl = euclidean(m,X);
eucl_res = (t~=out_eucl);


out_mahal = mahalanobis(m,S,X);
mahal_res = (t~=out_mahal);

error1 = nnz(eucl_res);
error2 = nnz(mahal_res);

fprintf("Error euclidean is %d percent\n", (error1*100)/N);
fprintf("Error mahalanobis is %d percent\n", (error2*100)/N);


[l,c]=size(m);

for i=1:N
	variable = X(:,i);
    for j=1:c
    	mean = m(:,j);
      exp_component = (variable - mean)'*(pinv(S))*(variable - mean);
      val(j) = (1/sqrt(2*pi)*det(S))*exp(-1/2*exp_component);
    end
    [num,z(i)]=max(val);
end


out_bayes = z;
bayes_res = (t~=out_bayes);
error3 = nnz(bayes_res);
fprintf("Error bayes is %d percent\n", (error3*100)/N);
