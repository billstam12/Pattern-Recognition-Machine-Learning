randn('seed',0) %Initialization of the random number generator for normal distr.
P1=0.5; %a priori class probabilities
P2=0.5;


m1=1; % mean of normal distributions
m2=4;

s=1; % variance of the normal distributions

N=10000; %Total number of points (Give only even numbers)

Y=[randn(1,N/2)+m1 randn(1,N/2)+m2];

t=[ones(1, N/2) 2*ones(1, N/2)]; 



output = [];

for i = 1:N
	p1=(1/(sqrt(2*pi)*s))*exp(-(Y(i)-m1)^2/(2*s));
	p2=(1/(sqrt(2*pi)*s))*exp(-(Y(i)-m2)^2/(2*s));

	if (P1*p1>P2*p2)
		output = [output 1];
	else 
		output = [output 2];
	end
end

bayes_res = (t~=output);

error  = nnz(bayes_res);
error1 = nnz(bayes_res(1:N/2));
error2 = nnz(bayes_res((N/2+1):N));

fprintf("Error is %d percent\n", (error1*100)/N);

fprintf("Error for class 1 is %d percent\n", (error1*100)/(N/2));
fprintf("Error for class 2 is %d percent\n", (error2*100)/(N/2));
