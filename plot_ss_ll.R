
library(tidyverse)
library(grid)
library(gridExtra)
library(png)
library(extrafont)


setwd('E:/Attention_discounting/attention_discount_project')

# -----------------
# Define functions
# -----------------

g <- function(t){
  1/(1-delta)*(delta^(-t)-1)
}

# weight function
w <- function(u,t){
  1 / (1+g(t)*exp(-u/lambda))
} 

# utility function
v <- function(x){
  x^gamma/lambda
}

# -----------------
# Set parameters
# -----------------

delta = 0.75
gamma = 0.6
lambda = 2
t_l = 16
t_s = 8
x_l = 30
x_s = 10
#x_l = 20
#x_s = 5


# -----------------
# Magnitude effect
# -----------------

# data for plot
t_seq = 0:20

u_l = v(x_l)
u_s = v(x_s)

ll_d <- tibble(w = w(u_l,t_seq), 
               reward = "x_l", 
               t = t_seq, 
               v = w(u_l,t_seq)*u_l)

ss_d <- tibble(w = w(u_s,t_seq), 
               reward = "x_s", 
               t = t_seq, 
               v = w(u_s,t_seq)*u_s)

d <- bind_rows(ll_d,ss_d)

# plot format
x_grid = 4
x_breaks = c(min(t_seq):(max(t_seq)/x_grid))*x_grid
new_labels_x <- c(expression(italic(x)[italic("T")]==30),
                  expression(italic(x)[italic("T")]==10))
    
# the declines of discounting factors
mag_effect <- ggplot(data = d,aes(x = t,y = w, color = reward))+
  geom_line()+
  theme_bw(11)+
  labs(color = 'reward level', 
       x = expression(italic("T")),
       y = expression(italic(w)[italic("T")]))+
  theme(text = element_text(family = "Times New Roman"),
        legend.position = c(0.8, 0.8))+
  scale_x_continuous(breaks = x_breaks)+
  scale_color_hue(breaks = c('x_l','x_s'), 
                  labels = new_labels_x)

mag_effect

path = 'E:/PhD Supervision/annual review/images'

ggsave(filename='mag_effect.png',path=path,width=10,height=8,units='cm')

# the value of reward sequence
ggplot(data = d,aes(x = t,y = v, color = reward))+
  geom_line()+
  theme_bw(11)+
  xlab("T")+
  ylab("Value of sequence")+
  ylim(c(0,4))+
  theme()+
  scale_x_continuous(breaks = x_breaks)


# --------------------------
# Common difference effect
# --------------------------

find_equiv <- function(u_l,u_s,t_s){
  t_seq_list = t_s+c(0:1000)*0.1
  obj <- abs(w(u_s,t_s)*u_s - w(u_l,t_seq_list)*u_l)
  eq_idx <- which(obj == min(obj))
  t_l <- t_seq_list[eq_idx]
  return(t_l)
}

u_l_low = v(30)
u_s_low = v(10)
u_l_high = v(120)
u_s_high = v(100)

t_l_equiv_1 <- sapply(t_seq, 
                      function(t_s) find_equiv(u_l=u_l_low,u_s=u_s_low,t_s))
t_l_equiv_2 <- sapply(t_seq, 
                      function(t_s) find_equiv(u_l=u_l_high,u_s=u_s_high,t_s))

df_common_diff_1 <- tibble(u_l = u_l_low,
                         u_s = u_s_low,
                         t_s = t_seq,
                         t_l_equiv = t_l_equiv_1,
                         cond = 'low_reward')

df_common_diff_2 <- tibble(u_l = u_l_high,
                           u_s = u_s_high,
                           t_s = t_seq,
                           t_l_equiv = t_l_equiv_2,
                           cond = 'high_reward')

d_com <- bind_rows(df_common_diff_1,df_common_diff_2)

new_labels_xx <- c(expression(italic(x)[italic(l)]==30~","~italic(x)[italic(s)]==10),
                  expression(italic(x)[italic(l)]==120~","~italic(x)[italic(s)]==100))

plot_com <- ggplot(data = d_com[d_com$t_s<11&d_com$t_s>0,],aes(x = t_s,y = t_l_equiv, color = cond))+
  geom_line()+
  geom_abline(slope=1,intercept = t_l_equiv_2[1]-0.3, linetype ='dashed')+
  theme_bw(11)+
  ylim(c(0,17))+
  labs(color = 'reward', 
       x = expression(italic(t)[italic(s)]),
       y = expression("equivalent long delay" ~~ italic(t)[italic(l)]))+
  theme(text = element_text(family = "Times New Roman"),
        legend.position = c(0.24, 0.82))+
  scale_x_continuous(breaks = c(1,3,5,7,9))+
  scale_color_hue(breaks = c('low_reward','high_reward'), 
                  labels = new_labels_xx)+
  annotate("text",x = 9.5,y = 12.8, size=3,
           label = paste0(45, "°"))

plot_com  
path = 'E:/PhD Supervision/annual review/images'
  
ggsave(filename='common_diff.png',path=path,width=10,height=8,units='cm') 

# -----------------
# Concavity of time discounting
# -----------------

t_conc = c(0:50)

xlarge <- tibble(w=w(v(300),t_conc), t=t_conc, reward = 'x_l')
xsmall <- tibble(w=w(v(30),t_conc), t=t_conc, reward = 'x_s')
dt_conc <- bind_rows(xlarge,xsmall)
new_labels <- c(expression(italic(x)[italic("T")]==300),
                expression(italic(x)[italic("T")]==30))
            
# discount function

plot_conc <- ggplot(data=dt_conc, aes(x=t,y=w, color=reward))+
  geom_line()+
  theme_bw(11)+
  labs(x=expression(italic("T")),
      y=expression(italic(w)[italic("T")]),
      color='reward level')+
  #ggtitle('(a) Convexity of discount function')+
  theme(legend.position = c(0.8, 0.8),
        text = element_text(family = "Times New Roman"))+
  scale_color_hue(breaks = c('x_l','x_s'), 
                  labels = new_labels)

plot_conc

path = 'E:/PhD Supervision/annual review/images'

ggsave(filename='concave_discount.png',path=path,width=10,height=8,units='cm')

# -----------------
# S-shaped value function
# -----------------

x_seq = c(1:300)
du_1 <- w(u=v(x_seq),t=t_l)* v(x_seq)
du_2 <- w(u=v(x_seq),t=t_s)* v(x_seq)

d1 <- tibble(du=du_1, x=x_seq, delay= 't_l')
d2 <- tibble(du=du_2, x=x_seq, delay= 't_s')
d_union <- bind_rows(d1,d2)
new_labels_t <- c(expression(italic("T")==16),
                expression(italic("T")==8))

# S-shaped value function

plot_s_value <- ggplot(d= d_union, aes(x = x,y = du, color=delay))+
          geom_line()+
          theme_bw(11)+
          ylab("value of sequence")+
          xlab(expression(italic(x)[italic("T")]))+
          #ggtitle('(b) S-shaped value function')+
          theme(legend.position = c(0.15, 0.8),
                text = element_text(family = "Times New Roman"))+
          scale_color_hue(breaks = c('t_l','t_s'), 
                          labels = new_labels_t)
plot_s_value

ggsave(filename='s_shape_value.png',path=path,width=10,height=8,units='cm')

#gridplot <- arrangeGrob(plot_conc,plot_s_value,ncol=2)
#path = 'rmarkdown/images/plot-discount-value.png'
#ggsave(path,gridplot,width=18,height=8,units='cm')


# -----------------
# Dynamic Inconsistency
# -----------------

att_choice <- read.csv('model/example_att_choice.csv')
inatt_choice <- read.csv('model/example_inatt_choice.csv')
a_labels <- c(expression(italic(t)==0),
              expression(italic(t)==1),
              expression(italic(t)==2),
              expression(italic(t)==3))

plot_att<- ggplot(d = att_choice, 
                 aes(x=period,y=consumption,color=as.factor(decision_step)))+
              geom_line()+
              theme_bw(11)+
              labs(color='decision step',x='period')+
              ggtitle('(a) with learning')+
              theme(legend.position = 'None',
                    text = element_text(family = "Times New Roman"))
  

plot_inatt<- ggplot(d = inatt_choice, 
                    aes(x=period,y=consumption,color=as.factor(decision_step)))+
                geom_line()+
                theme_bw(11)+
                labs(color='decision step',x = expression(italic(t)))+
                ggtitle('(b) without learning')+
                theme(text = element_text(family = "Times New Roman"))+
                scale_color_hue(breaks = as.factor(0:3), 
                                labels = a_labels)
    

gridplot2 <- arrangeGrob(plot_att,plot_inatt,ncol=2,widths=c(1, 1.3))

path = 'rmarkdown/images/plot-budget-dynamic.png'

ggsave(path,gridplot2,width=16.5,height=7,units='cm')




# ---- load data ----

img_T <- "E:/Attention_discounting/mydata/Plot_A=100_Y=70.png"
img_Y <- "E:/Attention_discounting/mydata/Plot_A=100_T=3.png"

img_T <- readPNG(img_T)
img_Y <- readPNG(img_Y)

grid_T <- rasterGrob(img_T, interpolate=TRUE)
grid_Y <- rasterGrob(img_Y, interpolate=TRUE)

subtitle_T <- textGrob("(a) Y=70, M=100", x=0.23, y=10.2,
                       gp = gpar(fontsize=8.5, fontfamily = "Times New Roman"))
subtitle_Y <- textGrob("(b) T=3, M=100", x=0.23, y =10.2,
                       gp = gpar(fontsize=8.5,fontfamily = "Times New Roman"))

grid_image1_with_subtitle <- arrangeGrob(grid_T, bottom = subtitle_T)
grid_image2_with_subtitle <- arrangeGrob(grid_Y, bottom = subtitle_Y)

pilot_show <- arrangeGrob(grid_image1_with_subtitle, grid_image2_with_subtitle, nrow = 2)


path = 'rmarkdown/images/pilot-result.png'

ggsave(path,pilot_show,width=6,height=10,units='cm')

