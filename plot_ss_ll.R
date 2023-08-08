
library(tidyverse)
library(gridExtra)

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
lambda = 1
t_l = 16
t_s = 8
x_l = 20
x_s = 5

# -----------------
# Common difference effect
# -----------------

# data for plot
t_seq = 0:t_l

u_l = v(x_l)
u_s = v(x_s)

ll_d <- tibble(w = w(u_l,t_seq), 
               reward = "LL", 
               t = t_seq, 
               v = w(u_l,t_seq)*u_l)

ss_d <- tibble(w = w(u_s,t_seq), 
               reward = "SS", 
               t = t_seq, 
               v = w(u_s,t_seq)*u_s)

d <- bind_rows(ll_d,ss_d)

# plot format
x_grid = 4
x_breaks = c(0:(t_l/x_grid))*x_grid


# the declines of discounting factors
ggplot(data = d,aes(x = t,y = w, color = reward))+
  geom_line()+
  theme_bw(11)+
  xlab("T")+
  ylab(expression(w[T]))+
  #theme(axis.title = element_text(face = 'italic'))+
  scale_x_continuous(breaks = x_breaks)


# the value of reward sequence
ggplot(data = d,aes(x = t,y = v, color = reward))+
  geom_line()+
  theme_bw(11)+
  xlab("T")+
  ylab("Value of reward sequence")+
  theme()+
  scale_x_continuous(breaks = x_breaks)


# -----------------
# Concavity of time discounting
# -----------------

t_conc = c(0:50)

xlarge <- tibble(w=w(v(x_l),t_conc), t=t_conc, reward = 'x_l')
xsmall <- tibble(w=w(v(x_s),t_conc), t=t_conc, reward = 'x_s')
dt_conc <- bind_rows(xlarge,xsmall)
new_labels <- c(expression(italic(x)[italic("T")]==20),
                expression(italic(x)[italic("T")]==5))
            
# discount function

plot_conc <- ggplot(data=dt_conc, aes(x=t,y=w, color=reward))+
  geom_line()+
  theme_bw(11)+
  xlab(expression(italic("T")))+
  ylab(expression(italic(w)[italic("T")]))+
  ggtitle('(a) Convexity of discount function')+
  theme(legend.position = c(0.85, 0.8),
        text = element_text(family = "Times New Roman"))+
  scale_color_hue(breaks = c('x_l','x_s'), 
                  labels = new_labels)


# -----------------
# S-shaped value function
# -----------------

x_seq = c(1:80)
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
          ylab("value of reward sequence")+
          xlab(expression(italic(x)[italic("T")]))+
          ggtitle('(b) S-shaped value function')+
          theme(legend.position = c(0.8, 0.2),
                text = element_text(family = "Times New Roman"))+
          scale_color_hue(breaks = c('t_l','t_s'), 
                          labels = new_labels_t)


gridplot <- arrangeGrob(plot_conc,plot_s_value,ncol=2)

path = 'rmarkdown/images/plot-discount-value.png'

ggsave(path,gridplot,width=18,height=8,units='cm')


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
              ggtitle('(a) Attentive decision maker')+
              theme(legend.position = 'None',
                    text = element_text(family = "Times New Roman"))
  

plot_inatt<- ggplot(d = inatt_choice, 
                    aes(x=period,y=consumption,color=as.factor(decision_step)))+
                geom_line()+
                theme_bw(11)+
                labs(color='decision step',x = expression(italic(t)))+
                ggtitle('(b) Inattentive decision maker')+
                theme(text = element_text(family = "Times New Roman"))+
                scale_color_hue(breaks = as.factor(0:3), 
                                labels = a_labels)
    

gridplot2 <- arrangeGrob(plot_att,plot_inatt,ncol=2,widths=c(1, 1.3))

path = 'rmarkdown/images/plot-budget-dynamic.png'

ggsave(path,gridplot2,width=16.5,height=7,units='cm')






