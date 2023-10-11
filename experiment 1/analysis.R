

library(readxl)
library(tidyverse)
library(ggpubr)
library(stringr)
library(lme4)
library(logistf)
library(optimx)

options(digits = 3)


coef_confint <- function(model, coef_name = NULL, conf_level = 0.95){
  
  critical <- qnorm((1 + conf_level) / 2)
  
  if(is.null(coef_name)){
    coef_est <- summary(model)[,"Estimate"]
    se <- summary(model)$coef[, "Std. Error"]
  }else{
    coef_est <- summary(model)$coef[coef_name,"Estimate"]
    se <- summary(model)$coef[coef_name, "Std. Error"]
  }

  conf_int <- data.frame(coefficient = coef_est,
                     std_error = se) %>%
              mutate(lower_bound = coefficient - critical*std_error,
                     upper_bound = coefficient + critical*std_error
                     )
  
  print(paste0('Confidence level:',conf_level))
  return(conf_int)
}


ara_solver <- function(x_risk,x_safe){
  a <- seq(0.001,0.2,by=0.001)
  u_risk <- 1-exp(-a*x_risk)
  u_safe <- 1-exp(-a*x_safe)
  diff <- abs(0.5*u_risk - u_safe)
  a[which.min(diff)]
}


# ----------------------------------
#         Load Data
# ----------------------------------
# choice data(N=160)
# Each choice is between option A and option B(labelled by 1 and 2) 
df_raw <- read_excel("experiment.1.data.xlsx",sheet = 1)[-1,]

df_raw$duration <- as.numeric(df_raw$`Duration (in seconds)`)
df_raw$prolific_id <- df_raw$PROLIFIC_PID

# Loop and Merge design for intertemporal choices 
lm_design <- read_excel("experiment.1.data.xlsx",sheet = 2)

# design for risky choices
risky_design <- read_excel("experiment.1.data.xlsx",sheet = 3)

# Filter data by attention check
# People should choose option B in Q7, and option A in Q8, and spend >3min 
cols <- colnames(df_raw)[29:249]
cols_attention_check_1 <- cols[grep("Q7", cols)]
cols_attention_check_2 <- cols[grep("Q8", cols)]

df_filtered <- df_raw %>% 
          select(c('prolific_id','duration',cols)) %>%
          mutate_at(vars(cols),as.numeric)%>%
          filter(if_all(cols_attention_check_1, ~ . == 2)) %>% 
          filter(if_all(cols_attention_check_2, ~ . == 1)) %>%
          filter(duration >180) %>%
          select(-c(cols_attention_check_1,cols_attention_check_2))

# 157 of the 160 participants pass attention check 
nrow(df_filtered)


# ----------------------------------
#   Characterize Utility Function
# ----------------------------------

# Gather data to make each choice occupy a row
# Split choice data into intertemporal choices and risky choices
df_choice <- df_filtered %>% 
  mutate(pid = factor(1:nrow(df_filtered))) %>%
  gather(key = 'question', value = 'choice', -c(pid, prolific_id,duration))

# risky choices: Q10, Q11, Q12
risky_cols <- cols[grep("Q10|Q11|Q12", cols)]

df_risky_choice <- df_choice %>%
  filter(question %in% risky_cols) %>%
  mutate(row_id = str_extract(question, "(?<=_)(\\d+)$"),
         q_id = str_extract(question, "(?<=Q)\\d+")) %>%
  mutate_at(vars(c(q_id,row_id)),as.numeric) %>% 
  group_by(prolific_id,pid,q_id,choice) %>%
  summarise(row_id = ifelse(unique(choice) == 1,max(row_id),min(row_id)))%>%
  left_join(risky_design) %>%
  group_by(prolific_id,pid,risk_amount) %>%
  summarise(safe_amount = mean(safe_amount))

ra_est <- df_risky_choice %>% 
  mutate(implied_rra = log(0.5,base = safe_amount/risk_amount),
         implied_ara = mapply(ara_solver,risk_amount/10,safe_amount/10))

# RRA yields smaller variance than ARA
ra_est %>% group_by(risk_amount) %>%
  summarise(rra_mean = mean(implied_rra),
            rra_sd = sd(implied_rra),
            ara_mean = mean(implied_ara)*10,
            ara_sd = sd(implied_ara)*10
            )

# mean rra: 0.749
crra <- mean(ra_est$implied_rra)


# intertemporal choices: Q5 (Immed_Rw_Vary), Q6 (Delayed_Rw_Vary)
df_time_choice <- df_choice %>%
  filter(!question %in% risky_cols) %>%
  mutate(q_id = str_extract(question, "\\d+(?=_)"),
         cond_id = str_extract(question, "(?<=Q)\\d+"),
         row_id = str_extract(question, "(?<=_)(\\d+)$")) %>%
  mutate_at(vars(c(q_id,cond_id,row_id)),as.numeric) %>%
  mutate(choice = choice -1,
         cond = ifelse(cond_id==5,'Delayed_Rw_Vary','Immed_Rw_Vary'))%>%
  left_join(lm_design, by = c('cond','q_id')) %>%
  mutate(b_vary_rw = row_id,
         b_fixed_rw = b_fixed_rw *0.1,
         a_rw = a_rw *0.1) %>%
  select(-c(q_id,cond_id,row_id,question))


# ----------------------------------
#       Baseline Model
# ----------------------------------

# Logitstic Regression:Immed_Rw_Vary

formula1 <- as.formula("choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + 
                b_vary_rw*factor(b_delay) + factor(pid)")

logit1 <- glm(formula1, 
            data = df_time_immed, 
            family = binomial(link='logit'))

coef_name <- names(logit1$coefficients)[-grep('pid',names(logit1$coefficients))]

fe_logit1 <- coef_confint(logit1,coef_name = coef_name)

# Logitstic Regression:Delayed_Rw_Vary

formula2 <- as.formula("choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + factor(pid)")

logit2 <- glm(formula2, 
              data = df_time_delayed, 
              family = binomial(link='logit'))

coef_name <- names(logit2$coefficients)[-grep('pid',names(logit2$coefficients))]

fe_logit2 <- coef_confint(logit2,coef_name = coef_name)


# # Generalized Linear Mixed Model (GLMM): use the Laplace approx. method to
# # to obtain likelihood function, the result is similar to regression with
# # dummy variables (but slightly different).
# 
# # Immed_Rw_Vary
# 
# mod1 <- glmer(choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) +
#                 b_vary_rw*factor(b_delay) + (1|pid),
#               data = df_time_immed,
#               family = binomial(link='logit'),
#               control = glmerControl(optimizer = "bobyqa",calc.derivs = TRUE))
# 
# # Gradient information
# mod1@optinfo$derivs
# 
# # Coefficients
# summary(mod1)
# confint(mod1, method = "Wald")
# 
# 
# # Delayed_Rw_Vary
# 
# mod2 <- glmer(choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + (1|pid),
#               data = df_time_delayed,
#               family = binomial(link='logit'),
#               control = glmerControl(optimizer = "bobyqa",calc.derivs = TRUE))
# 
# # Gradient information
# mod2@optinfo$derivs
# 
# # Coefficients
# summary(mod2)
# confint(mod2, method = "Wald")


# ----------------------------------
#          Utility Model
# ----------------------------------

# Immed_Rw_Vary

df_time_immed$u_a_rw <- (df_time_immed$a_rw*10)^crra/10
df_time_immed$u_b_vary_rw <- (df_time_immed$b_vary_rw*10)^crra/10

formula_u1 <- as.formula("choice ~ u_a_rw + u_b_vary_rw*factor(b_fixed_rw) + 
                u_b_vary_rw*factor(b_delay)+ factor(pid)")

logit_u1 <- glm(formula_u1, 
                data = df_time_immed, 
                family = binomial(link='logit'))

coef_name <- names(logit_u1$coefficients)[-grep('pid',names(logit_u1$coefficients))]

fe_logit_u1 <- coef_confint(logit_u1,coef_name = coef_name)

fe_logit_u1


# Delayed_Rw_Vary

df_time_delayed$u_a_rw <- (df_time_delayed$a_rw*10)^crra/10
df_time_delayed$u_b_vary_rw <- (df_time_delayed$b_vary_rw*10)^crra/10

formula_u2 <- as.formula("choice ~ u_a_rw + u_b_vary_rw*factor(b_fixed_rw) + 
                + factor(pid)")

logit_u2 <- glm(formula_u2, 
              data = df_time_delayed, 
              family = binomial(link='logit'))

coef_name <- names(logit_u2$coefficients)[-grep('pid',names(logit_u2$coefficients))]

fe_logit_u2 <- coef_confint(logit_u2,coef_name = coef_name)

fe_logit_u2

# Mapping reward amount to utility does not change fitting performance

# ----------------------------------
#       Bias-Reduced Model
# ----------------------------------

# Firth's Bias-Reduced Regression: Immed_Rw_Vary
#sample_id <- sample(unique(df_time_immed$pid),20)
#sample <- df_time_immed[df_time_immed$pid %in% sample_id,]

firth1 <- logistf(formula1, 
                  data = df_time_immed, 
                  family = binomial(link='logit'))

save(firth1, file = "firth1.RData")

# Firth's Bias-Reduced Regression: Delayed_Rw_Vary
#sample_id <- sample(unique(df_time_delayed$pid),20)
#sample <- df_time_delayed[df_time_delayed$pid %in% sample_id,]

firth2 <- logistf(formula2, 
                  data = df_time_delayed, 
                  family = binomial(link='logit'))

save(firth2, file = "firth2.RData")


