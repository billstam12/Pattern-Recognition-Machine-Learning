m1 = 1;
m2 = 3;
s =1;

randn('seed',0);
N1 = 200;
N2 = 200;
fprintf("Producing arrays with %d random elements for %d and %d random elements for %d\n", N1,m1,N2,m2);

X1 = randn(1,N1)+m1;
X2 = randn(1,N2)+m2;

randn('seed',0);

N = 1000;
fprintf("Producing array with %d random elements for %d and %d random elements for %d\n ", N1,m1,N2,m2);
Y = [randn(1,N/2)+m1 randn(1,N/2)+m2];
t = [ones(1,N/2) 2*ones(1,N/2)];

P1 = N1/(N1+N2);
P2 = N2/(N1+N2);


%Bayes rule for the case where the true means are used
output=[];
for i=1:N
	p1=(1/(sqrt(2*pi)*s))*exp(-(Y(i)-m1)^2/(2*s));
	p2=(1/(sqrt(2*pi)*s))*exp(-(Y(i)-m2)^2/(2*s));
	% Application of the Bayes rule
	if(P1*p1>P2*p2)
		output=[output 1];
	else
		output=[output 2];
	end
end
bayes_res=(t~=output); %if bayes_res(i)=1 then the i-th point is correctly classified

error1 = nnz(bayes_res);

fprintf("Error for bayes method is %f\n",(error1*100)/N);

output_Parzen=[];
h=0.2;
for i=1:N
	p1 = sum (abs((X1- Y(i))/h)<=1/2)/(N*h);
	p2 = sum (abs((X2- Y(i))/h)<=1/2)/(N*h);
	if(P1*p1>P2*p2)
		output_Parzen=[output_Parzen 1];
 	else
 		output_Parzen=[output_Parzen 2];
 	end
end

bayes_res_parzen = (t~=output_Parzen);


error2 = nnz(bayes_res_parzen);

fprintf("Error for parzen windows method is %f\n",(error2*100)/N);










