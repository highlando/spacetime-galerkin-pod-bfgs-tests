function f_red=fNewton_PODDEIM(yOld_red,y_red,f_red, control_red)

global Bred Fred Cred nu T Nt h Mred1

dt= T/(Nt-1); 

f_red = (1/dt)*y_red - (1/dt)*yOld_red + 0.5*Bred*(Fred*y_red).^2 + nu*Cred*y_red - h*f_red - Mred1*control_red;

end