close all
clear all
clf 

addpath('./FOM');
% addpath('./POD');
% addpath('./PODDEIM');

global nu tol_newton max_newton h dt omega M B C Ns Nt N T L time space plotOn

flag = 2;

if flag == 1 
    addpath('./SPG');
elseif flag == 2
    addpath('./BFGS','./BFGS/immoptibox/');
elseif flag == 3
    addpath('./Newton-type');
end


N  = 219;                 % # grid point in space
Ns = 120;                 % # grid point in time
Nt = Ns;
L  = 1;                   % x \in [0,L]
T  = 1;                   % t \in [0,T]

x = linspace(0,L,N+1);           % Spacial discretization
t = linspace(0,T,Nt);            % time discretization
[time, space] = meshgrid(t',x);  % for 3D plotting
h = L/N;                         % constant step size
dt= T/(Nt-1);                    % constant time step

tol_newton = 10e-8;     % for Newton's method
max_newton = 50;        % max. number of inner Newton iterations

plotOn = 1;             % plotting and output on (1) or off(0)


nu_test    = [0.005, 0.001];
alpha_test = [0.001, 0.0001, 0.0];

res_jan = [0.17325481, 0.17965117;
           0.17280781, 0.17920417;
           0.17275814, 0.1791545];

for ii = 3:3
    for jj = 1:1

        omega = alpha_test(ii);   % control penalty
        nu    = nu_test(jj);      % viscosity parameter     

        %% Initialize the state y_0
        y0 = [ones(floor((N-1)/2),1);zeros(ceil((N-1)/2),1)];
%         y0 = [zeros(floor((N-1)/2),1);ones(ceil((N-1)/2),1)];

        %% desired state z, does not depend on time
        z  = y0;

        %% Pre-compute stiffness and mass matrix for full-order Burgers eqn
        M = (h/6)*gallery('tridiag',ones(N-2,1),4*ones(N-1,1),ones(N-2,1));
        B = gallery('tridiag',-0.5*ones(N-2,1),zeros(N-1,1),0.5*ones(N-2,1));
        C = (1/h)*gallery('tridiag',-ones(N-2,1),2*ones(N-1,1),-ones(N-2,1));

                
        %% Solve Burgers eqn with control u0
        u0 = zeros((N-1)*Nt,1);
        Y = Burgers(reshape(u0,N-1,Nt),y0);


        %% Check results
        Z  = kron(ones(1,Nt),z);
        u1 = 0*z+1;
        U1 = kron(ones(1,Nt),u1);
%         disp(['nu = ', num2str(nu), '  alpha = ', num2str(omega)])
%         error    = L2norm(Y,Z);


        Ybc  = [zeros(1,Nt);Y;zeros(1,Nt)];
        Zbc  = [zeros(1,Nt);Z;zeros(1,Nt)];
        U1bc = [ones(1,Nt);U1;ones(1,Nt)];
        
        
        cost_fun = 0.5*L2norm(Ybc,Zbc)^2 + omega/2*L2norm(U1bc,0*U1bc)^2
        error    = L2norm(Ybc,Zbc)
        
        
        abs( res_jan(ii,jj) - cost_fun)
    end
end

if plotOn == 1
    figure(1)
    mesh(time, space, Ybc);
    zlim([0 1.5])
    title('solution')
    
    figure(2)
    mesh(time, space, Zbc);
    zlim([0 1.5])
    title('target')
    
    figure(3)
    mesh(time, space, Ybc-Zbc);
    title('error')
end