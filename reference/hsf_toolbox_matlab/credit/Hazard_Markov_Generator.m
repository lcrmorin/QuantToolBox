function lambda = Density_Markov_Generator(t, Lambda)
S = Survival_Markov_Generator(t,Lambda);
f = Density_Markov_Generator(t,Lambda);
lambda = f./S;
end
