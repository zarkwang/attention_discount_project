
# ----------------------------------
# Bias-Reduced Model:Immed_Rw_Vary
# ----------------------------------

library(lme4)
library(logistf)

df_choice <- read.csv('intertemporal_choice_obs.csv')[,-1]

df_time_delayed <- df_choice[df_choice$cond == 'Delayed_Rw_Vary',]


# Firth's Bias-Reduced Regression 
sample_id <- sample(unique(df_time_delayed$pid),10)
sample <- df_time_delayed[df_time_delayed$pid %in% sample_id,]

formula2 <- as.formula("choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + factor(pid)")

firth_reg2 <- logistf(formula2, 
                      data = sample, 
                      family = binomial(link='logit'),
                      )


save(firth_reg2, file = "firth_result2.RData")


