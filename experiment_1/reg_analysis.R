
library(readxl)
library(tidyverse)
library(stringr)
library(lme4)
#library(optimx)


options(digits = 3)

# ----------------------------------
#         Useful Functions
# ----------------------------------

coef_confint <- function(model, coef_name = NULL, conf_level = 0.95){
  
  critical <- qnorm((1 + conf_level) / 2)
  
  if(is.null(coef_name)){
    coef_est <- summary(model)[,"Estimate"]
    se <- summary(model)$coef[, "Std. Error"]
    p_value <- summary(model)$coef[, "Pr(>|z|)"]
  }else{
    coef_est <- summary(model)$coef[coef_name,"Estimate"]
    se <- summary(model)$coef[coef_name, "Std. Error"]
    p_value <- summary(model)$coef[coef_name, "Pr(>|z|)"]
  }

  conf_int <- data.frame(coef = coef_est,
                         bse = se,
                         p_value = p_value) %>%
              mutate(lower_bound = coef - critical*bse,
                     upper_bound = coef + critical*bse
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
df_choice <- df_filtered %>% 
  mutate(pid = factor(1:nrow(df_filtered))) %>%
  gather(key = 'question', value = 'choice', -c(pid, prolific_id,duration))


# Split choice data into intertemporal choices and risky choices
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


# ----------------------------------
#      Descriptive Analysis
# ----------------------------------

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


# plot the variance of choice for each question
df_time_equiv <- df_time_choice %>%
  group_by(prolific_id,pid,cond,a_rw,b_fixed_rw,b_delay,choice) %>%
  summarise(b_vary_rw = ifelse(unique(choice) == 0,max(b_vary_rw),min(b_vary_rw)))%>%
  group_by(prolific_id,pid,cond,a_rw,b_fixed_rw,b_delay) %>%
  summarise(b_vary_rw = mean(b_vary_rw))


sum_time_equiv <- df_time_equiv %>% 
  group_by(cond,a_rw,b_fixed_rw,b_delay) %>%
  summarise(mean_vary_rw = mean(b_vary_rw),
            std_vary_rw = sd(b_vary_rw))

ggplot(data=sum_time_equiv,
       aes(x=factor(b_fixed_rw),
           y=std_vary_rw))+
  geom_point(aes(shape= factor(b_delay),
                 color= factor(a_rw)),
             size = 2)+
  geom_line(aes(group = interaction(b_delay, a_rw)),
            linetype = 'dashed', color = 'grey') +
  facet_wrap(~cond,
             labeller = as_labeller(
               c('Immed_Rw_Vary' = 'Immediate reward varies',
                 'Delayed_Rw_Vary' = 'Delayed reward varies')))+
  labs(x = 'reward amount constant across rows in B (×£10)',
       y = 'standard deviation') +
  scale_shape_discrete(name = "Time length of B (month)") +
  scale_color_discrete(name = "Reward amount of A (×£10)") +
  theme_bw(12)+
  theme(
    legend.position = 'top',
    legend.key.width = unit(0.3, "cm"),
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    text = element_text(family = "Times New Roman")
  )

ggsave('./figures/fig_switch_sd.png',device = 'png',width = 18, height = 9, units = 'cm')



# construct datasets for regression
df_time_immed <- df_time_choice[df_time_choice$cond == 'Immed_Rw_Vary',]
df_time_delayed <- df_time_choice[df_time_choice$cond == 'Delayed_Rw_Vary',]

# ----------------------------------
#       Baseline Model
# ----------------------------------
# Logitstic Regression:Immed_Rw_Vary

formula1 <- as.formula("choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + 
                b_vary_rw*factor(b_delay) + factor(pid)")

logit1 <- glm(formula1, 
            data = df_time_immed, 
            family = binomial(link='logit'))

coef_name1 <- names(logit1$coefficients)[-grep('pid',names(logit1$coefficients))]

fe_logit1 <- coef_confint(logit1,coef_name = coef_name1)

df_time_immed$pred_logit <- predict(logit1,type='response')

# Logitstic Regression:Delayed_Rw_Vary

formula2 <- as.formula("choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + factor(pid)")

logit2 <- glm(formula2, 
              data = df_time_delayed, 
              family = binomial(link='logit'))

coef_name2 <- names(logit2$coefficients)[-grep('pid',names(logit2$coefficients))]

fe_logit2 <- coef_confint(logit2,coef_name = coef_name2)

df_time_delayed$pred_logit <- predict(logit2,type='response')


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

print('Baseline model fitted')

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

coef_name_u1 <- names(logit_u1$coefficients)[-grep('pid',names(logit_u1$coefficients))]

fe_logit_u1 <- coef_confint(logit_u1,coef_name = coef_name_u1)

df_time_immed$pred_logit_u <- predict(logit_u1,type='response')

# Delayed_Rw_Vary

df_time_delayed$u_a_rw <- (df_time_delayed$a_rw*10)^crra/10
df_time_delayed$u_b_vary_rw <- (df_time_delayed$b_vary_rw*10)^crra/10

formula_u2 <- as.formula("choice ~ u_a_rw + u_b_vary_rw*factor(b_fixed_rw) + 
                + factor(pid)")

logit_u2 <- glm(formula_u2, 
              data = df_time_delayed, 
              family = binomial(link='logit'))

coef_name_u2 <- names(logit_u2$coefficients)[-grep('pid',names(logit_u2$coefficients))]

fe_logit_u2 <- coef_confint(logit_u2,coef_name = coef_name_u2)

df_time_delayed$pred_logit_u <- predict(logit_u2,type='response')

# Mapping reward amount to utility does not change fitting performance

print('Utility model fitted')

# ------------------------------------------
#   Utility + Eliminating Uniform Choices
# ------------------------------------------
# Immed_Rw_Vary
q_filter1 <- df_time_immed %>% 
  group_by(a_rw,b_fixed_rw,b_delay,b_vary_rw) %>%
  summarise(mean_choice = mean(choice)) %>%
  filter(mean_choice >0 & mean_choice <1)

df_censor_immed <- df_time_immed %>% inner_join(q_filter1)

logit_c1 <- glm(formula_u1, 
                data = df_censor_immed,
                family = binomial(link='logit'))

fe_logit_c1 <- coef_confint(logit_c1,coef_name = coef_name_u1)


# Delayed_Rw_Vary
q_filter2 <- df_time_delayed %>% 
  group_by(a_rw,b_fixed_rw,b_vary_rw) %>%
  summarise(mean_choice = mean(choice)) %>%
  filter(mean_choice >0 & mean_choice <1)

df_censor_delayed <- df_time_delayed %>% inner_join(q_filter2)

logit_c2 <- glm(formula_u2, 
                data = df_censor_delayed,
                family = binomial(link='logit'))

fe_logit_c2 <- coef_confint(logit_c2,coef_name = coef_name_u2)

print('Utility model without uniform choices fitted')


# ------------------------------------------
#     Utility + Option A as Treatment
# ------------------------------------------
# Logitstic Regression:Immed_Rw_Vary

formula_a1 <- as.formula("choice ~ u_b_vary_rw*factor(a_rw) + 
                u_b_vary_rw*factor(b_fixed_rw) + 
                u_b_vary_rw*factor(b_delay) + factor(pid)")

logit_a1 <- glm(formula_a1, 
                data = df_time_immed, 
                family = binomial(link='logit'))

coef_name_a1 <- names(logit_a1$coefficients)[-grep('pid',names(logit_a1$coefficients))]

fe_logit_a1 <- coef_confint(logit_a1,coef_name = coef_name_a1)

df_time_immed$pred_logit_a <- predict(logit_a1,type='response')

# Logitstic Regression:Delayed_Rw_Vary

formula_a2 <- as.formula("choice ~ u_b_vary_rw*factor(a_rw) + 
                       u_b_vary_rw*factor(b_fixed_rw) + factor(pid)")

logit_a2 <- glm(formula_a2, 
                data = df_time_delayed, 
                family = binomial(link='logit'))

coef_name_a2 <- names(logit_a2$coefficients)[-grep('pid',names(logit_a2$coefficients))]

fe_logit_a2 <- coef_confint(logit_a2,coef_name = coef_name_a2)

df_time_delayed$pred_logit_a <- predict(logit_a2,type='response')

print('Utility model with option A treatment fitted')

# ------------------------------------------
#             Save Results
# ------------------------------------------

write.csv(rbind(df_time_immed,df_time_delayed),file='intertemporal_choice_obs.csv')


