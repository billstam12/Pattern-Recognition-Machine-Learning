function [z] = bayes(m,Y)
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

error = nnz(~bayes_res);