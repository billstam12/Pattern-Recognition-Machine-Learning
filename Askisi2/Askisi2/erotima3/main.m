close('all');
clear;

m(:,1) = [0 0]';
m(:,2) = [4 4]';

S1 = eye(2);
S2 = eye(2);
N1 = 200;

randn('seed',0)
X1=[mvnrnd(m(:,1),S1,170); mvnrnd(m(:,2),S2,30)]';
z1=[ones(1,170) 2*ones(1,30)];

N2=200;

randn('seed',100)
X2=[mvnrnd(m(:,1),S1,170); mvnrnd(m(:,2),S2,30)]';
z2=[ones(1,170) 2*ones(1,30)];


% 2. Augment the data vectors of X1 
X1=[X1; ones(1,sum(N1))];
y1=2*z1-3;

% Augment the data vectors of X2
X2=[X2; ones(1,sum(N2))];
y2=2*z2-3;

% Compute the classification error of the LS classifier based on X2
[w]=SSErr(X1,y1,0)
SSE_out=2*(w'*X2>0)-1;
err_SSE=sum(SSE_out.*y2<0)/sum(N2)
figure(1), plot(X1(1,y1==1),X1(2,y1==1),'bo',...
X1(1,y1==-1),X1(2,y1==-1),'r.')
hold on;
figure(1), axis equal

xp = linspace(min(X1), max(X1), 100);
yp = - (w(1)*xp + w(3))/w(2);
plot(xp, yp);
hold off;
