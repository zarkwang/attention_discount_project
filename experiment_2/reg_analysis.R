
setwd('E:/Attention_discounting/experiment_2')

library('tidyverse')
library('lme4')

df_time_valid <- read.csv('valid_sequence_data.csv')


df_choice_vary <- df_time_valid %>% 
  select(worker_id,value_surplus) %>%
  mutate(total_money = value_surplus == 60) %>%
  group_by(worker_id) %>% 
  summarize(sum_total_money = sum(total_money),
            sd_value = sd(value_surplus)) 

df_choice_vary <-dplyr::select(df_time_valid, worker_id, value_surplus) %>%
  dplyr::mutate(total_money = value_surplus == 60) %>%
  dplyr::group_by(worker_id) %>% 
  dplyr::summarize(sum_total_money = sum(total_money),
                   sd_value = sd(value_surplus)) 

id_list <- df_choice_vary[df_choice_vary$sd_value > 0,]$worker_id

id_list <- df_choice_vary[df_choice_vary$sum_total_money < 14,]$worker_id

df_time_valid <- df_time_valid[df_time_valid$worker_id %in% id_list,]

df_filtered <- df_time_valid[(df_time_valid$value_surplus > 0)&
                               (df_time_valid$value_surplus < 60),]


df_filtered <- df_time_valid[(df_time_valid$value_surplus > sort(df_time_valid$value_surplus)[6])&
                               (df_time_valid$value_surplus < sort(df_time_valid$value_surplus,decreasing = TRUE)[6]),]

model <- lm(value_surplus ~ front_amount:seq_length + factor(worker_id), 
              data = df_filtered)


summary(model)

confint(model)

ggplot(data=df_choice_vary)+
  geom_histogram(aes(x=sd_value))
  


library('MASS')


model <- rlm(value_surplus ~ front_amount:seq_length:factor(label) + factor(worker_id), 
            data = df_time_label, maxit=100)

summary(model)

confint.default(model)

sort(df_time_valid$value_surplus,decreasing=T)[1:15]
sort(df_time_valid$value_surplus)[1:15]

12 / nrow(df_time_valid)








setwd('D:/attention/attention_discount_project/experiment_2')

df_time_label <- read.csv('labeled_result.csv')

df_label_filtered <- df_time_label[(df_time_label$value_surplus > sort(df_time_label$value_surplus)[6])&
                               (df_time_label$value_surplus < sort(df_time_label$value_surplus,decreasing = TRUE)[6]),]


model <- lm(value_surplus ~ front_amount:seq_length:factor(label) + factor(worker_id), 
            data = df_label_filtered)


summary(model)

confint(model)

# front-end amount increases by 100, average value of back-end amount decrease
# 1.5 - 2.8 when delay = 12 months
# 1.2 - 2.5 when delay = 6 months



