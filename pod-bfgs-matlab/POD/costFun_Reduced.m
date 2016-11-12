function f = costFun_Reduced( u_red, Y_red, z_red )

global h T k3 Nt omega Mred2

dt = T/(Nt-1);

U_red = reshape(u_red,k3,Nt);

f = 0;
for t=1:Nt
    f = f + dt*(0.5*Y_red(:,t)'*Y_red(:,t) - h*z_red'*Y_red(:,t) + omega/2*U_red(:,t)'*Mred2*U_red(:,t)); 
end

end