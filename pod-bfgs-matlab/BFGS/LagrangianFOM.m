function f = LagrangianFOM( u, Y, z )

global T Ns omega M h N

dt= T/(Ns-1); 

U = reshape(u,N-1,Ns);

f = 0;
for t=1:Ns
    f = f + dt*(0.5*Y(:,t)'*M*Y(:,t) - h*z'*Y(:,t) + omega/2*U(:,t)'*M*U(:,t)); 
end


