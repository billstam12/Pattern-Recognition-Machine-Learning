randn('seed',0) %Initialization of the random number generator for normal distr.
P1=0.5; %a priori class probabilities
P2=0.5;


m1=1; % mean of normal distributions
m2=4;

s=1; % variance of the normal distributions







p1= (1/(sqrt(2*pi)*s))*exp(-(1.7-m1)^2/(2*s));
p2=(1/(sqrt(2*pi)*s))*exp(-(1.7-m2)^2/(2*s));

if (P1*p1>P2*p2)
	fprintf("MPLAAA\n");
else 
	fprintf("2\n");
end

