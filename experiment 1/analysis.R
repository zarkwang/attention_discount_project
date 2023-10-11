
setwd("E:/Attention_discounting/mydata")

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


# ----------------------------------
#         Load Data
# ----------------------------------
# choice data(N=160)
# Each choice is between option A and option B(labelled by 1 and 2) 
df_raw <- read_excel("experiment.1.data.xlsx",sheet = 1)[-1,]

df_raw$duration <- as.numeric(df_raw$`Duration (in seconds)`)
df_raw$prolific_id <- df_raw$PROLIFIC_PID

# Loop and Merge design 
lm_design <- read_excel("experiment.1.data.xlsx",sheet = 2)


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

# Split choice data into intertemporal choices and risky choices
# intertemporal choices: Q5, Q6
# risky choices: Q10, Q11, Q12
df_choice <- df_filtered %>% 
  mutate(pid = factor(1:nrow(df_filtered))) %>%
  gather(key = 'question', value = 'choice', -c(pid, prolific_id,duration))

risky_cols <- cols[grep("Q10|Q11|Q12", cols)]

df_risky_choice <- df_choice %>%
  filter(question %in% risky_cols)

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

dummies <- model.matrix(~ pid - 1, data = df_time_choice)

df_time_choice <- cbind(df_time_choice,dummies)


# ----------------------------------
#       Baseline Model
# ----------------------------------

# Logitstic Regression:Immed_Rw_Vary

df_time_immed <- df_time_choice[df_time_choice$cond == 'Immed_Rw_Vary',]

formula1 <- paste("choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + 
                b_vary_rw*factor(b_delay) +", 
                paste(colnames(dummies), collapse = " + "))

logit1 <- glm(as.formula(formula1), 
            data = df_time_immed, 
            family = binomial(link='logit'))

coef_name <- names(logit1$coefficients)[-grep('pid',names(logit1$coefficients))]

fe_logit1 <- coef_confint(logit1,coef_name = coef_name)


# Logitstic Regression:Delayed_Rw_Vary

df_time_delayed <- df_time_choice[df_time_choice$cond == 'Delayed_Rw_Vary',]

formula2 <- paste("choice ~ a_rw + b_vary_rw*factor(b_fixed_rw) + ", 
                  paste(colnames(dummies), collapse = " + "))

logit2 <- glm(as.formula(formula2), 
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
# df_time_delayed <- df_time_choice[df_time_choice$cond == 'Delayed_Rw_Vary',]
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
#       Bias-Reduced Model
# ----------------------------------

# Firth's Bias-Reduced Regression: Immed_Rw_Vary
firth1 <- logistf(as.formula(formula1), 
                  data = df_time_immed, 
                  family = binomial(link='logit'))













# Plots: Immed_Rw_Vary

df_time_immed$choice_pred <- predict(mod1, type='response')

tab_immed <- df_time_immed %>%
            group_by(b_vary_rw,b_fixed_rw,b_delay,a_rw) %>%
            summarise(mean_choice = mean(choice),
                      mean_pred = mean(choice_pred)) %>%
            mutate(b_vary_rw = b_vary_rw *10,
                   b_fixed_rw = b_fixed_rw *10,
                   a_rw = a_rw *10)

a_rw_list <- unique(tab_immed$a_rw)


fig_immed_1 <- ggplot(data = tab_immed[tab_immed$a_rw == a_rw_list[1],],
                      aes(x = b_vary_rw,
                          y = mean_choice,
                          color = factor(b_fixed_rw), 
                          shape = factor(b_delay))) +
  geom_point(size=2)+
  geom_line(aes(y=mean_pred),linetype = 'dotdash')+
  ggtitle(paste0('(a) option A: £',a_rw_list[1], ''))+
  labs(x = "immediate reward in B (£)", 
       y = "proportion of choosing B")+
  scale_x_continuous(breaks = c(1:10)*10) +
  scale_shape_discrete(name = "time length of B (month)") +
  scale_color_discrete(name = "delayed reward in B (£)") +
  theme_bw(12)+
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.spacing.y = unit(-0.2, "cm"),
    legend.key.width = unit(0.3, "cm"),
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    text = element_text(family = "Times New Roman")
  )


fig_immed_2 <- ggplot(data = tab_immed[tab_immed$a_rw == a_rw_list[2],],
                      aes(x = b_vary_rw, 
                          y = mean_choice, 
                          color = factor(b_fixed_rw), 
                          shape = factor(b_delay))) +
  geom_point(size=2)+
  geom_line(aes(y=mean_pred),linetype='dotdash')+
  ggtitle(paste0('(b) option A: £',a_rw_list[2], ''))+
  labs(x = "immediate reward in B (£)", 
       y = "proportion of choosing B")+
  scale_x_continuous(breaks = c(1:10)*10) +
  scale_shape_discrete(name = "time length of B (month)") +
  scale_color_discrete(name = "delayed reward in B (£)") +
  theme_bw(12)+
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.spacing.y = unit(-0.2, "cm"),
    legend.key.width = unit(0.3, "cm"),
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    text = element_text(family = "Times New Roman")
  )


ggarrange(fig_immed_1, fig_immed_2, 
          ncol = 2, common.legend = TRUE, legend="bottom")

ggsave('fig_immed_vary.png',device = 'png',width = 18, height = 10, units = 'cm')


# Plots: Delayed_Rw_Vary

# tab_delayed <- df_time_delayed %>%
#   group_by(b_vary_rw,b_fixed_rw,a_rw) %>%
#   summarise(mean_choice = mean(choice)) %>%
#   mutate(b_vary_rw = b_vary_rw *10,
#          b_fixed_rw = b_fixed_rw *10,
#          a_rw = a_rw *10)



tab_delayed <- df_time_delayed %>%
  group_by(b_vary_rw,b_fixed_rw,a_rw) %>%
  summarise(mean_choice = mean(choice)) %>%
  filter(mean_choice >0 & mean_choice <1) %>%
  mutate(b_vary_rw = b_vary_rw *10,
         b_fixed_rw = b_fixed_rw *10,
         a_rw = a_rw *10,
         log_odds = log(mean_choice / (1-mean_choice)))


fig_delayed_1 <- ggplot(data = tab_delayed[tab_delayed$a_rw == a_rw_list[1],],
                      aes(x = b_vary_rw, 
                          y = log_odds, 
                          color = factor(b_fixed_rw))) +
  geom_point(size=2)+
  geom_line()+
  ggtitle(paste0('(a) option A: £',a_rw_list[1], ''))+
  labs(x = "delayed reward in option B (£)", 
       y = "proportion of choosing B")+
  scale_x_continuous(breaks = c(1:10)*10) +
  scale_color_discrete(name = "immediate reward in option B (£)") +
  theme_bw(9)+
  theme(
    legend.position = "bottom",
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    text = element_text(family = "Times New Roman")
  )


fig_delayed_2 <- ggplot(data = tab_delayed[tab_delayed$a_rw == a_rw_list[2],],
                        aes(x = b_vary_rw, 
                            y = log_odds, 
                            color = factor(b_fixed_rw))) +
  geom_point(size=2)+
  geom_line()+
  ggtitle(paste0('(a) option A: £',a_rw_list[2], ''))+
  labs(x = "delayed reward in option B(£)", 
       y = "proportion of choosing B")+
  scale_x_continuous(breaks = c(1:10)*10) +
  scale_color_discrete(name = "immediate reward in option B (£)") +
  theme_bw(9)+
  theme(
    legend.position = "bottom",
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    text = element_text(family = "Times New Roman")
  )


ggarrange(fig_delayed_1, fig_delayed_2, 
          ncol = 2, common.legend = TRUE, legend="bottom")

ggsave('fig_delayed_vary.png',device = 'png',width = 12, height = 7, units = 'cm')





