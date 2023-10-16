
library(tidyverse)
library(ggpubr)
library(gridExtra)

source('reg_analysis.R')


# ----------------------------------
#       Plot Baseline Results
# ----------------------------------
# Plots: Immed_Rw_Vary
firth_pred_immed <- read.csv('firth_pred_immed.csv')[,-1]

tab_immed <- df_time_immed %>%
  group_by(b_vary_rw,a_rw,b_fixed_rw,b_delay) %>%
  summarise(mean_choice = mean(choice),
            pred_logit = mean(pred_logit),
            pred_logit_u = mean(pred_logit_u)
            ) %>%
  left_join(firth_pred_immed) %>%
  mutate(b_vary_rw = b_vary_rw*10,
         b_fixed_rw = b_fixed_rw*10,
         a_rw = a_rw*10)


fig_immed_1 <- ggplot(data = tab_immed[tab_immed$a_rw == unique(tab_immed$a_rw)[1],],
                      aes(x = b_vary_rw,
                          y = mean_choice,
                          color = factor(b_fixed_rw), 
                          shape = factor(b_delay))) +
  geom_point(size=2)+
  geom_line(aes(y=pred_logit),linetype = 'dashed',alpha=0.75)+
  ggtitle(paste0('(1) option A: £',unique(tab_immed$a_rw)[1], ''))+
  labs(x = "immediate reward in option B (£)", 
       y = "probability of choosing B")+
  scale_x_continuous(breaks = c(1:10)*10) +
  scale_shape_discrete(name = "time length of option B (month)") +
  scale_color_discrete(name = "delayed reward in option B (£)") +
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


fig_immed_2 <- ggplot(data = tab_immed[tab_immed$a_rw == unique(tab_immed$a_rw)[2],],
                      aes(x = b_vary_rw, 
                          y = mean_choice, 
                          color = factor(b_fixed_rw), 
                          shape = factor(b_delay))) +
  geom_point(size=2)+
  geom_line(aes(y=pred_logit),linetype='dashed',alpha=0.75)+
  ggtitle(paste0('(2) option A: £',unique(tab_immed$a_rw)[2], ''))+
  labs(x = "immediate reward in option B (£)", 
       y = "")+
  scale_x_continuous(breaks = c(1:10)*10) +
  scale_shape_discrete(name = "time length of option B (month)") +
  scale_color_discrete(name = "delayed reward in option B (£)") +
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


panel_immed <- ggarrange(fig_immed_1, fig_immed_2, 
          ncol = 2, common.legend = TRUE, legend="bottom")


# Plots: Delayed_Rw_Vary
firth_pred_delayed <- read.csv('firth_pred_delayed.csv')[,-1]

tab_delayed <- df_time_delayed %>%
  group_by(b_vary_rw,a_rw,b_fixed_rw,b_delay) %>%
  summarise(mean_choice = mean(choice),
            pred_logit = mean(pred_logit),
            pred_logit_u = mean(pred_logit_u)) %>%
  left_join(firth_pred_delayed) %>%
  mutate(b_vary_rw = b_vary_rw*10,
         b_fixed_rw = b_fixed_rw*10,
         a_rw = a_rw*10)


fig_delayed_1 <- ggplot(data = tab_delayed[tab_delayed$a_rw == unique(tab_delayed$a_rw)[1],],
                        aes(x = b_vary_rw, 
                            y = mean_choice, 
                            color = factor(b_fixed_rw))) +
  geom_point(size=2)+
  geom_line(aes(y=pred_logit),linetype='dashed',alpha=0.75)+
  ggtitle(paste0('(1) option A: £',unique(tab_delayed$a_rw)[1], ''))+
  labs(x = "delayed reward in option B (£)", 
       y = "probability of choosing B")+
  scale_x_continuous(breaks = c(1:10)*10) +
  scale_color_discrete(name = "immediate reward in option B (£)") +
  theme_bw(12)+
  theme(
    legend.position = "bottom",
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    text = element_text(family = "Times New Roman")
  )


fig_delayed_2 <- ggplot(data = tab_delayed[tab_delayed$a_rw == unique(tab_delayed$a_rw)[2],],
                        aes(x = b_vary_rw, 
                            y = mean_choice, 
                            color = factor(b_fixed_rw))) +
  geom_point(size=2)+
  geom_line(aes(y=pred_logit),linetype='dashed',alpha=0.75)+
  ggtitle(paste0('(2) option A: £',unique(tab_delayed$a_rw)[2], ''))+
  labs(x = "delayed reward in option B (£)", 
       y = "")+
  scale_x_continuous(breaks = c(1:10)*10) +
  scale_color_discrete(name = "immediate reward in option B (£)") +
  theme_bw(12)+
  theme(
    legend.position = "bottom",
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    text = element_text(family = "Times New Roman")
  )


panel_delayed <- ggarrange(fig_delayed_1, fig_delayed_2, 
          ncol = 2, common.legend = TRUE, legend="bottom")


title_immed <- grid::textGrob("Panel A: Immediate reward varies",
                         gp = grid::gpar(fontsize=14, fontfamily = "Times New Roman"))
title_delayed <- grid::textGrob("Panel B: Delayed reward varies",
                        gp = grid::gpar(fontsize=14, fontfamily = "Times New Roman"))

panel_immed_with_title <- arrangeGrob(title_immed, panel_immed,
                                      heights = c(0.2,1))
panel_delayed_with_title <- arrangeGrob(title_delayed, panel_delayed,
                                        heights = c(0.2,1))

fig_grand <- grid.arrange(panel_immed_with_title,
                      panel_delayed_with_title, 
                      ncol =1,
                      heights = c(1.15,1))

ggsave('./figures/fig_grand_baseline.png',fig_grand,device = 'png',width = 18, height = 20, units = 'cm')


# ----------------------------------
#     Robustness Check Tables
# ----------------------------------



