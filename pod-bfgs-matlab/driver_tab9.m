clear all
clf
close all
warning off
format short

n_exp = 5;
Khat  = [24, 36, 48, 72, 96];
Khat  = Khat/4;
Jtarget = [0.0702 0.0617 0.0468 0.0354 0.0304]/2;

erom       = zeros(length(Khat),n_exp);
Jrom       = zeros(length(Khat),n_exp);
bfgs_iters = zeros(length(Khat),n_exp);
trom       = zeros(length(Khat),n_exp);

%% Table 9 -- POD
for i=1:n_exp
    [erom(:,i), Jrom(:,i), bfgs_iters(:,i), trom(:,i)] = exper_bfgs(Khat, Jtarget, 0);
end

% pick best run
for i=1:length(Khat)
   [trom_best,ind] = min(trom(i,:));
    
   disp(['POD red order = ', num2str(Khat(i))])
   e = 0.5*erom(i,ind);
   Jopt = Jrom(i,ind);
   num_iters = bfgs_iters(i,ind);
   walltime = trom(i,ind);
   
   disp(['$', num2str(e), '$'])
   disp(['$', num2str(Jopt), '$'])
   disp(['$', num2str(num_iters), '$'])
   disp(['\texttt{', num2str(walltime), '}'])
   
   disp(['----------------------------------------------'])
end


%% Table 9 -- POD-DEIM
for i=1:n_exp
    [erom(:,i), Jrom(:,i), bfgs_iters(:,i), trom(:,i)] = exper_bfgs(Khat, Jtarget, 1);
end
 
% pick best run
for i=1:length(Khat)
   [trom_best,ind] = min(trom(i,:));
    
   disp(['POD DEIM red order = ', num2str(Khat(i))])
   
   e = 0.5*erom(i,ind);
   Jopt = Jrom(i,ind);
   num_iters = bfgs_iters(i,ind);
   walltime = trom(i,ind);
   
   disp(['$', num2str(e), '$'])
   disp(['$', num2str(Jopt), '$'])
   disp(['$', num2str(num_iters), '$'])
   disp(['\texttt{', num2str(walltime), '}'])

   disp(['----------------------------------------------'])
end

