# Import relevant packages
library(plyr)
library(emmeans)
library(ggplot2)
library(tidyr)
library(ggstatsplot)
library(lmerTest)
library(performance)        
library(see)                 
library(effectsize)            
library(sjPlot)
library(effects)
library(interactions)
library(dplyr)
getwd()
# dataset, simulations of social learning --------------
simulations_emu_vs_copy <- read.csv('simulations/results_emu_vs_copy_2025-01-20_18-47-42.csv') #Import data
simulations_emu_vs_rl_imit <- read.csv('simulations/results_emu_vs_rl_imit_2025-01-20_18-27-40.csv') #Import data

simulations_emu_vs_rl_imit$normalized_reward_diff <- (simulations_emu_vs_rl_imit$model_based_reward - simulations_emu_vs_rl_imit$model_free_reward)/simulations_emu_vs_rl_imit$model_free_reward
simulations_emu_vs_copy$normalized_reward_diff <- (simulations_emu_vs_copy$model_based_reward - simulations_emu_vs_copy$copy_last_reward)/simulations_emu_vs_copy$copy_last_reward
# Aggregate data across all parameters
agg_rl <- simulations_emu_vs_rl_imit %>%
  group_by(volatility, Uncertainty) %>%
  summarise(mean_reward_diff = mean(normalized_reward_diff))
agg_rl$mean_reward_diff
# Create first plot
p1_all <- ggplot(agg_rl, aes(x = volatility, y = mean_reward_diff, color = Uncertainty)) +
  geom_point(size = 2) +
  geom_line() +
  scale_color_viridis_d(option = "plasma", direction = 1, begin = 0.05, end = 0.5) +
  labs(title = "Emulation vs RL Imitation",
       x = "Volatility",
       y = "Normalized reward difference") +
  theme_minimal()+
  theme(
    legend.position = "none",
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    plot.title = element_text(size = 16)
  )+xlim(0.05,0.3)+ylim(0.1,0.25)
p1_all
# Second plot - emu vs copy without optimization
# Aggregate data across all parameters
agg_copy_all <- simulations_emu_vs_copy %>%
  group_by(volatility, Uncertainty) %>%
  summarise(mean_reward_diff = mean(normalized_reward_diff))

# Create second plot
p2_all <- ggplot(agg_copy_all, aes(x = volatility, y = mean_reward_diff, color = Uncertainty)) +
  geom_point(size = 2) +
  geom_line() +
  scale_color_viridis_d(option = "plasma", direction = 1, begin = 0.05, end = 0.5) +
  labs(title = "Emulation vs 1-step Imitation", 
       x = "Volatility",
       y = "Normalized reward difference") +
  theme_minimal()+
  theme(
    legend.position = "bottom",
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    plot.title = element_text(size = 16)
  )+xlim(0.05,0.3)+ylim(0.0,0.5)

# Combine plots using patchwork
p1_all / p2_all
# save fig 
ggsave("output/Figure_2.png", width = 6, height = 8, dpi = 800)
