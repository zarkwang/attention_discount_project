

# ----------------------------------
# Bias-Reduced Model:Immed_Rw_Vary
# ----------------------------------

library(logistf)

df_choice <- read.csv('intertemporal_choice_obs.csv')[,-1]

df_time_immed <- df_choice[df_choice$cond == 'Immed_Rw_Vary',]


# Firth's Bias-Reduced Regression 
sample_id <- sample(unique(df_time_immed$pid),10)
sample <- df_time_immed[df_time_immed$pid %in% sample_id,]

formula1 <- as.formula("choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + 
                b_vary_rw*factor(b_delay) + factor(pid)")

firth_reg1 <- logistf(formula1, 
                  data = sample, 
                  family = binomial(link='logit'))

save(firth_reg1, file = "firth_result1.RData")

