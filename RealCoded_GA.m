% This is a MATLAB implementation of real-coded GA algorithm[1][2]
% [1] K. Deb, A. Kumar. Real-coded genetic algorithms 
% with simulated binary crossover: studies on multimodal and multiobjective 
% problems.Complex Systems, 1995, 9(6):431-54.
% [2] K. Deb. An efficient constraint handling method
% for genetic algorithms. Computer Methods in Applied Mechanics and Engineering
% , 2000, 186(2):311-38. 
%------------------------------------
% parent selection: 2-tournament selection
% crossover: simulated binary crossover
% mutation: ploynomial mutation
% using environment selection
%------------------------------------
clearvars;clc;close all;
%------------------------------------
% problem name 'Fun_Ellipsoid','Fun_Rosenbrock', 'Fun_Ackley', 'Fun_Griewank'
obj_fun = 'Fun_Ackley';
% number of variablese
num_vari = 50;
% population size of genetic algorithm
pop_size = 100;
% maximum number of generations
max_gen = 500;
%------------------------------------
% design space
switch obj_fun
    case 'Fun_Ellipsoid'
        lower_bound = -5.12*ones(1,num_vari); upper_bound = 5.12*ones(1,num_vari);
    case 'Fun_Rosenbrock'
        lower_bound = -2.048*ones(1,num_vari); upper_bound = 2.048*ones(1,num_vari);
     case 'Fun_Ackley'   
         lower_bound = -32.768*ones(1,num_vari); upper_bound = 32.768*ones(1,num_vari);
    case 'Fun_Griewank'
         lower_bound = -600*ones(1,num_vari); upper_bound = 600*ones(1,num_vari);
end
% best objectives in each generation
best_obj_record = zeros(max_gen,1);
% the first generation
generation = 1;
% generate random population
pop_vari = repmat(lower_bound,pop_size,1) + rand(pop_size, num_vari).*repmat((upper_bound-lower_bound),pop_size,1);
% calculate the objective values
pop_fitness = feval(obj_fun, pop_vari);
best_obj_record(generation,:) = min(pop_fitness);
% print the iteration information
plot(best_obj_record(1:generation));
xlabel('generation');ylabel('best objective value');
title(sprintf('GA on %d-d %s function\n generation: %d, best: %0.4g',num_vari,obj_fun(5:end),generation, best_obj_record(generation,:)));drawnow;
fprintf('GA on %s, generation: %d, evaluation: %d, best: %0.4g\n',obj_fun(5:end),generation,generation*pop_size,best_obj_record(generation,:))
%------------------------------------
% the evoluation of the generation
while generation < max_gen
    %------------------------------------
    % parent selection using k-tournament (default k=2) selection
    k = 2;
    temp = randi(pop_size,pop_size,k);
    [~,index] = min(pop_fitness(temp),[],2);
    pop_parent = pop_vari(sum(temp.*(index == 1:k),2),:);
    %------------------------------------
    % crossover (simulated binary crossover)
    % dic_c is the distribution index of crossover 
    dis_c = 1;
    mu  = rand(pop_size/2,num_vari);
    parent1 = pop_parent(1:2:pop_size,:);
    parent2 = pop_parent(2:2:pop_size,:);
    beta = 1 + 2*min(min(parent1,parent2)-lower_bound,upper_bound-max(parent1,parent2))./max(abs(parent2-parent1),1E-6);
    alpha = 2 - beta.^(-dis_c-1);
    betaq = (alpha.*mu).^(1/(dis_c+1)).*(mu <= 1./alpha) + (1./(2-alpha.*mu)).^(1/(dis_c+1)).*(mu > 1./alpha);
    % the mutation is performed randomly on each variable
    betaq = betaq.*(-1).^randi([0,1],pop_size/2,num_vari);
    betaq(rand(pop_size/2,num_vari)>0.5) = 1;
    offspring1 = 0.5*((1+betaq).*parent1 + (1-betaq).*parent2);
    offspring2 = 0.5*((1-betaq).*parent1 + (1+betaq).*parent2);
    pop_crossover = [offspring1;offspring2];
    %------------------------------------
    % mutation (ploynomial mutation)
    % dis_m is the distribution index of polynomial mutation
    dis_m = 1;
    pro_m = 1/num_vari;
    rand_var = rand(pop_size,num_vari);
    mu  = rand(pop_size,num_vari);
    deta = min(pop_crossover-lower_bound, upper_bound-pop_crossover)./(upper_bound-lower_bound);
    detaq = zeros(pop_size,num_vari);
    position1 = rand_var<=pro_m & mu<=0.5;
    position2 = rand_var<=pro_m & mu>0.5;
    detaq(position1) = ((2*mu(position1) + (1-2*mu(position1)).*(1-deta(position1)).^(dis_m+1)).^(1/(dis_m+1))-1); 
    detaq(position2) = (1 - (2*(1-mu(position2))+2*(mu(position2)-0.5).*(1-deta(position2)).^(dis_m+1)).^(1/(dis_m+1)));
    pop_mutation = pop_crossover + detaq.*(upper_bound-lower_bound);
    %------------------------------------
    % fitness calculation
    pop_mutation_fitness = feval(obj_fun, pop_mutation);
    %------------------------------------
    % environment selection
    pop_vari_iter = [pop_vari;pop_mutation];
    pop_fitness_iter = [pop_fitness;pop_mutation_fitness];
    [~,win_num] = sort(pop_fitness_iter);
    pop_vari = pop_vari_iter(win_num(1:pop_size),:);
    pop_fitness = pop_fitness_iter(win_num(1:pop_size),:);  
    %------------------------------------
    % update the evaluation number of generation number
    generation = generation + 1;
    best_obj_record(generation,:) = min(pop_fitness);
    % print the iteration information
    plot(best_obj_record(1:generation));
    xlabel('generation');ylabel('best objective value');
    title(sprintf('GA on %d-d %s function\n generation: %d, best: %0.4g',num_vari,obj_fun(5:end),generation, best_obj_record(generation,:)));drawnow;
    fprintf('GA on %s, generation: %d, evaluation: %d, best: %0.4g\n',obj_fun(5:end),generation,generation*pop_size,best_obj_record(generation,:))
end

