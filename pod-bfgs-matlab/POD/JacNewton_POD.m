function J_red=JacNewton_POD(y)

global Cred nu T Nt k1

dt= T/(Nt-1); 

J_red = (1/dt)*eye(k1) + Ntildey(y) + nu*Cred;

end