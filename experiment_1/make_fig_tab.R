
library(tidyverse)
library(ggpubr)
library(gridExtra)
library(xtable)

source('reg_analysis.R')

write_document = FALSE

# ----------------------------------
#         Useful Functions
# ----------------------------------

star <- function(p){
  if(p < 0.001){
    star = '$^{***}$'
  } else if(p < 0.01){
    star = '$^{**}$'
  } else if(p < 0.05){
    star = '$^{*}$'
  } else {
    star = ''
  }
  return(star)
}

make_ci_tab <- function(data, add_var_name = FALSE){
  
  tab <- data['coef']
  
  tab$Coef <- paste0(formatC(data$coef, digits = 3, format = "f"),
                     mapply(star,data$p_value))
  
  tab$`95\\% CI` <- paste0('[',formatC(data$lower_bound, digits=3, format="f"),
                           ', ',formatC(data$upper_bound, digits=3, format="f"),
                           ']')
  if(add_var_name == TRUE){
    tab$var_name = rownames(tab)
  }
  
  return(tab[,-1])
}


write_reg_tab <- function(filename,
                          tab,var_name = NULL,
                          document = FALSE,
                          output = TRUE,
                          addtorow = NULL){
  
  begin_doc <- "%\n\\documentclass[12pt]{article}\n \\begin{document}"
  end_doc <- '\\end{document}'
  
  if(!is.null(var_name)){
    rownames(tab) <- var_name
  }
  
  table <- xtable(tab)
  align(table) <- c('l',rep('c',ncol(tab)))
  
  reg_table <-print.xtable(table, 
                           #hline.after = c(-1, 0, nrow(tab)),
                           #table.placement = "h",
                           floating = FALSE,
                           sanitize.text.function = function(x) {x},
                           include.colnames = FALSE,
                           add.to.row = addtorow)
  if(document == TRUE){
    reg_table <- paste0(begin_doc,str_split(reg_table, '\n', n = 2)[[1]][2],end_doc)
  }
  
  if(output == TRUE){
    write(reg_table, file = filename)
  }
}


# ----------------------------------
#         Baseline Table
# ----------------------------------
var_name_immed <- c('$M$',
                    '$X_v$',
                    '$\\textbf{1}\\{X_f = 7\\}$',
                    '$\\textbf{1}\\{X_f = 9\\}$',
                    '$\\textbf{1}\\{T = 9\\}$',
                    '$\\textbf{1}\\{T = 18\\}$',
                    '$X_v\\cdot\\textbf{1}\\{X_f = 7\\}$',
                    '$X_v\\cdot\\textbf{1}\\{X_f = 9\\}$',
                    '$X_v\\cdot\\textbf{1}\\{T = 9\\}$',
                    '$X_v\\cdot\\textbf{1}\\{T = 18\\}$')

var_name_delayed <- c('$M$',
                      '$X_v$',
                      '$\\textbf{1}\\{X_f = 7\\}$',
                      '$\\textbf{1}\\{X_f = 9\\}$',
                      '$X_v\\cdot\\textbf{1}\\{X_f = 7\\}$',
                      '$X_v\\cdot\\textbf{1}\\{X_f = 9\\}$')


# Panel A: Immed_Rw_Vary
firth_result_immed <- read.csv('firth_result_immed.csv')[,-1]
tab1 <- make_ci_tab(fe_logit1)[-1,]
tab2 <- make_ci_tab(firth_result_immed)
tab_baseline_a <- cbind(tab1,tab2)
remove(tab1,tab2)

row_mod <- sprintf(" & \\multicolumn{2}{c}{%s} & \\multicolumn{2}{c}{%s} \\\\",
                   "(1) Logit model",
                   "(2) Firth's model")
row_name <- "& Coef & 95\\% CI & Coef & 95\\% CI \\\\"
row_obs <- sprintf("\\hline observations & \\multicolumn{2}{c}{%d} & \\multicolumn{2}{c}{%d} \\\\",
                   nrow(df_time_immed),
                   nrow(df_time_immed))


addtorow <- list()
addtorow$pos <- list(0,0,nrow(tab_baseline_a))
addtorow$command  <- as.vector(c(row_mod,row_name,row_obs),mode='character')

write_reg_tab('./tables/baseline_A.tex',tab_baseline_a,
              var_name=var_name_immed,
              document=write_document,
              addtorow = addtorow)

# Panel B: Delayed_Rw_Vary
firth_result_delayed <- read.csv('firth_result_delayed.csv')[,-1]
tab1 <- make_ci_tab(fe_logit2)[-1,]
tab2 <- make_ci_tab(firth_result_delayed)
tab_baseline_b <- cbind(tab1,tab2)
remove(tab1,tab2)

row_obs <- sprintf("\\hline observations & \\multicolumn{2}{c}{%d} & \\multicolumn{2}{c}{%d} \\\\",
                   nrow(df_time_delayed),
                   nrow(df_time_delayed))

addtorow <- list()
addtorow$pos <- list(0,0,nrow(tab_baseline_b))
addtorow$command  <- as.vector(c(row_mod,row_name,row_obs),mode='character')

write_reg_tab('./tables/baseline_B.tex',tab_baseline_b,
              var_name=var_name_delayed,
              document=write_document,
              addtorow = addtorow)



# ----------------------------------
#     Table for Utility Models
# ----------------------------------
var_name_immed_u <- c('$u(M)$',
                      '$u(X_v)$',
                      '$\\textbf{1}\\{X_f = 7\\}$',
                      '$\\textbf{1}\\{X_f = 9\\}$',
                      '$\\textbf{1}\\{T = 9\\}$',
                      '$\\textbf{1}\\{T = 18\\}$',
                      '$u(X_v)\\cdot\\textbf{1}\\{X_f = 7\\}$',
                      '$u(X_v)\\cdot\\textbf{1}\\{X_f = 9\\}$',
                      '$u(X_v)\\cdot\\textbf{1}\\{T = 9\\}$',
                      '$u(X_v)\\cdot\\textbf{1}\\{T = 18\\}$',
                      '$\\textbf{1}\\{M = 12\\}$',
                      '$u(X_v)\\cdot\\textbf{1}\\{M = 12\\}$')

var_name_delayed_u <- c('$u(M)$',
                      '$u(X_v)$',
                      '$\\textbf{1}\\{X_f = 7\\}$',
                      '$\\textbf{1}\\{X_f = 9\\}$',
                      '$u(X_v)\\cdot\\textbf{1}\\{X_f = 7\\}$',
                      '$u(X_v)\\cdot\\textbf{1}\\{X_f = 9\\}$',
                      '$\\textbf{1}\\{M = 12\\}$',
                      '$u(X_v)\\cdot\\textbf{1}\\{M = 12\\}$')

# Panel A: Immed_Rw_Vary
tab1 <- make_ci_tab(fe_logit_u1,add_var_name = TRUE)[-1,]
tab2 <- make_ci_tab(fe_logit_c1,add_var_name = TRUE)[-1,]
tab3 <- make_ci_tab(fe_logit_a1,add_var_name = TRUE)[-1,]

union_rows <- data.frame(var_name = union(rownames(tab2),rownames(tab3)))

tab1_adj <- left_join(union_rows, tab1, by = "var_name")[,-1]
tab2_adj <- left_join(union_rows, tab2, by = "var_name")[,-1]
tab3_adj <- left_join(union_rows, tab3, by = "var_name")[,-1]

tab_mod_a <- cbind(tab1_adj,tab2_adj,tab3_adj)
rownames(tab_mod_a) <- var_name_immed_u
remove(tab1,tab2,tab3,tab1_adj,tab2_adj,tab3_adj)


row_mod <- sprintf(" & \\multicolumn{2}{c}{%s} & \\multicolumn{2}{c}{%s} & \\multicolumn{2}{c}{%s} \\\\",
                   "(3) Utility model",
                   "(4) Censored data",
                   "(5) Add interation")

row_name <- "& Coef & 95\\% CI & Coef & 95\\% CI & Coef & 95\\% CI \\\\"
row_obs <- sprintf("\\hline observations & \\multicolumn{2}{c}{%d} & \\multicolumn{2}{c}{%d} & \\multicolumn{2}{c}{%d} \\\\",
                   nrow(df_time_immed),
                   nrow(df_censor_immed),
                   nrow(df_time_immed))
row_aic <- sprintf("AIC & \\multicolumn{2}{c}{%.2f} & \\multicolumn{2}{c}{%.2f} & \\multicolumn{2}{c}{%.2f} \\\\",
                   logit_u1$aic,
                   logit_c1$aic,
                   logit_a1$aic)
  

addtorow <- list()
addtorow$pos <- list(0,0,nrow(tab_mod_a),nrow(tab_mod_a))
addtorow$command  <- as.vector(c(row_mod,row_name,row_obs,row_aic),mode='character')

write_reg_tab('./tables/utility_A.tex',tab_mod_a,
              var_name=var_name_immed_u,
              document=write_document,
              addtorow = addtorow)


# Panel A: Delayed_Rw_Vary
tab1 <- make_ci_tab(fe_logit_u2,add_var_name = TRUE)[-1,]
tab2 <- make_ci_tab(fe_logit_c2,add_var_name = TRUE)[-1,]
tab3 <- make_ci_tab(fe_logit_a2,add_var_name = TRUE)[-1,]

union_rows <- data.frame(var_name = union(rownames(tab2),rownames(tab3)))

tab1_adj <- left_join(union_rows, tab1, by = "var_name")[,-1]
tab2_adj <- left_join(union_rows, tab2, by = "var_name")[,-1]
tab3_adj <- left_join(union_rows, tab3, by = "var_name")[,-1]

tab_mod_b <- cbind(tab1_adj,tab2_adj,tab3_adj)
rownames(tab_mod_b) <- var_name_delayed_u
remove(tab1,tab2,tab3,tab1_adj,tab2_adj,tab3_adj)

row_mod <- sprintf(" & \\multicolumn{2}{c}{%s} & \\multicolumn{2}{c}{%s} & \\multicolumn{2}{c}{%s} \\\\",
                   "(3) Utility model",
                   "(4) Censored data",
                   "(5) Add interation")

row_name <- "& Coef & 95\\% CI & Coef & 95\\% CI & Coef & 95\\% CI \\\\"
row_obs <- sprintf("\\hline observations & \\multicolumn{2}{c}{%d} & \\multicolumn{2}{c}{%d} & \\multicolumn{2}{c}{%d} \\\\",
                   nrow(df_time_delayed),
                   nrow(df_censor_delayed),
                   nrow(df_time_delayed))
row_aic <- sprintf("AIC & \\multicolumn{2}{c}{%.2f} & \\multicolumn{2}{c}{%.2f} & \\multicolumn{2}{c}{%.2f} \\\\",
                   logit_u2$aic,
                   logit_c2$aic,
                   logit_a2$aic)


addtorow <- list()
addtorow$pos <- list(0,0,nrow(tab_mod_b),nrow(tab_mod_b))
addtorow$command  <- as.vector(c(row_mod,row_name,row_obs,row_aic),mode='character')

write_reg_tab('./tables/utility_B.tex',tab_mod_b,
              var_name=var_name_delayed_u,
              document=write_document,
              addtorow = addtorow)

# ----------------------------------
#     Plot Prediction Results
# ----------------------------------
# This figure is drawn based on the utility model with option A
# set as treatment

# Plots: Immed_Rw_Vary
firth_pred_immed <- read.csv('firth_pred_immed.csv')[,-1]

tab_immed <- df_time_immed %>%
  group_by(b_vary_rw,a_rw,b_fixed_rw,b_delay) %>%
  summarise(mean_choice = mean(choice),
            pred_logit = mean(pred_logit),
            pred_logit_u = mean(pred_logit_u),
            pred_logit_a = mean(pred_logit_a)
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
  geom_line(aes(y=pred_logit_a),linetype = 'dashed',alpha=0.75)+
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
  geom_line(aes(y=pred_logit_a),linetype='dashed',alpha=0.75)+
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
            pred_logit_u = mean(pred_logit_u),
            pred_logit_a = mean(pred_logit_a)) %>%
  left_join(firth_pred_delayed) %>%
  mutate(b_vary_rw = b_vary_rw*10,
         b_fixed_rw = b_fixed_rw*10,
         a_rw = a_rw*10)


fig_delayed_1 <- ggplot(data = tab_delayed[tab_delayed$a_rw == unique(tab_delayed$a_rw)[1],],
                        aes(x = b_vary_rw, 
                            y = mean_choice, 
                            color = factor(b_fixed_rw))) +
  geom_point(size=2)+
  geom_line(aes(y=pred_logit_a),linetype='dashed',alpha=0.75)+
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
  geom_line(aes(y=pred_logit_a),linetype='dashed',alpha=0.75)+
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

ggsave('./figures/fig_grand_pred.png',fig_grand,device = 'png',width = 18, height = 20, units = 'cm')









