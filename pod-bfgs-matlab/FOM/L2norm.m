function eval = L2norm( F, G )

global h T Nt

dt= T/(Nt-1); 


nx = size(F,1)-1;
nt = size(F,2)-1;

eval = 0.0;
for t=1:nt
    for i=1:nx
%         eval = eval + dt*h*( F(i,t) - G(i,t) )^2;
        eval = eval + (h*dt/4)*( (F(i,t) - G(i,t))^2 ...
                               + (F(i+1,t) - G(i+1,t))^2 ...
                               + (F(i,t+1) - G(i,t+1))^2 ...
                               + (F(i+1,t+1) - G(i+1,t+1))^2 );
    end
end
eval = sqrt(eval);
end

