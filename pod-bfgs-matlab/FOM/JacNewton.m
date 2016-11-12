function J=JacNewton(y)

global M C nu T Ns

dt= T/(Ns-1); 

J = (1/dt)*M + Ny(y) + nu*C;
end