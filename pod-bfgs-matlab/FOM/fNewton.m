function eval = fNewton(Yold,y,f,control)

global h M B C nu Ns T

dt= T/(Ns-1); 

eval = (1/dt)*M*y - (1/dt)*M*Yold + 0.5*B*y.^2 + nu*C*y - h*f - M*control;

end