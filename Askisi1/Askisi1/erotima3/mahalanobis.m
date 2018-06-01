function [z] = mahalanobis(m,S,X)

[l,c] = size(m);
[l,N] = size(X);

for i=1:N
	for j=1:c
		de(j) = sqrt((X(:,i) - m(:,j))' * pinv(S)*(X(:,i) - m(:,j)));
	end
	[l,z(i)] = min(de);
end