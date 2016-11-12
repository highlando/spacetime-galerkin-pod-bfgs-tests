clear all
clf
close all

warning off
format short

n_exp   = 5;
qhat    = [18, 17, 16, 14, 12, 10, 8];
nt      = [6, 7, 8, 10, 12, 14, 16];
Jtarget = 0.0468/2*ones(length(qhat));

erom       = zeros(length(qhat),n_exp);
Jrom       = zeros(length(qhat),n_exp);
bfgs_iters = zeros(length(qhat),n_exp);
trom       = zeros(length(qhat),n_exp);

%% Table 10
for i=1:n_exp
    [erom(:,i), Jrom(:,i), bfgs_iters(:,i), trom(:,i)] = exper_var_space_time(qhat, nt, Jtarget, 0);
end

% pick best run
for i=1:length(qhat)
   [trom_best,ind] = min(trom(i,:));
    
   disp(['q = ', num2str(qhat(i)), ', nt = ', num2str(nt(i))])
   
   e = 0.5*erom(i,ind);
   Jopt = Jrom(i,ind);
   num_iters = bfgs_iters(i,ind);
   walltime = trom(i,ind);
   
   disp(['$', num2str(e), '$'])
   disp(['$', num2str(Jopt), '$'])
   disp(['$', num2str(num_iters), '$'])
   disp(['$', num2str(walltime), '$'])
   
   disp(['----------------------------------------------'])
end