%SCRIPT


randn('seed',0) %Initialization of the randn function
m=[0 0]';
N=500;

%iteration

fprintf("Printing results for a\n");
S = initializeS(1,0,1);
X = mvnrnd(m,S,N)';

figure(1), plot(X(1,:), X(2,:),'.');
figure(1), axis equal;
figure(1), axis([-7 7 -7 7]);

pause;
fprintf("Printing results for b\n");

S = initializeS(0.2,0,0.2);
X = mvnrnd(m,S,N)';

figure(1), plot(X(1,:), X(2,:),'.');
figure(1), axis equal;
figure(1), axis([-7 7 -7 7]);

pause;

S = initializeS(2,0,2);
X = mvnrnd(m,S,N)';
fprintf("Printing results for c\n");

figure(1), plot(X(1,:), X(2,:),'.');
figure(1), axis equal;
figure(1), axis([-7 7 -7 7]);

pause;

S = initializeS(0.2,0,2);
X = mvnrnd(m,S,N)';
fprintf("Printing results for d\n");

figure(1), plot(X(1,:), X(2,:),'.');
figure(1), axis equal;
figure(1), axis([-7 7 -7 7]);

pause;

S = initializeS(2,0,0.2);
X = mvnrnd(m,S,N)';
fprintf("Printing results for e\n");

figure(1), plot(X(1,:), X(2,:),'.');
figure(1), axis equal;
figure(1), axis([-7 7 -7 7]);

pause;

S = initializeS(1,0.5,1);
X = mvnrnd(m,S,N)';
fprintf("Printing results for f\n");

figure(1), plot(X(1,:), X(2,:),'.');
figure(1), axis equal;
figure(1), axis([-7 7 -7 7]);

pause;

S = initializeS(0.3,0.5,2);
X = mvnrnd(m,S,N)';
fprintf("Printing results for g\n");

figure(1), plot(X(1,:), X(2,:),'.');
figure(1), axis equal;
figure(1), axis([-7 7 -7 7]);

pause;

S = initializeS(0.3,-0.5,2);
X = mvnrnd(m,S,N)';
fprintf("Printing results for h\n");

figure(1), plot(X(1,:), X(2,:),'.');
figure(1), axis equal;
figure(1), axis([-7 7 -7 7]);

pause;