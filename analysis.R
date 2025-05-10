# Import relevant packages
library(plyr)
library(dplyr)
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
library(glmmTMB)  
library(DHARMa)  
library(parameters)
library(brms)
library(bayesplot)

# set wd to the current folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# First dataset for behavioral analysis of critical trials
df_concat = read.csv("data/processed/unified_data_2025-01-16_00.csv")
str(df_concat)
unique(df_concat$response)
# aggregate data by participant
agg_data = ddply(df_concat, .(participant_id), summarise,
                 accuracy = mean(accurate),
                 n_comprehension_check_repetitions = mean(comprehension_check_repetitions))
agg_data

#exclude participants based on preregistered exclusion criteria
Ss_to_exclude_accuracy = agg_data$participant_id[agg_data$accuracy < 0.45]
Ss_to_exclude_comprehension_check_repetitions = agg_data$participant_id[agg_data$n_comprehension_check_repetitions > 2]
Ss_to_exclude_comprehension_check_repetitions
Ss_to_exclude_accuracy
#exclude participants based on exclusion criteria
Ss_to_exclude = unique(c(Ss_to_exclude_accuracy, Ss_to_exclude_comprehension_check_repetitions))
Ss_to_exclude
#exclude participants from df_concat
df_concat = df_concat[!df_concat$participant_id %in% Ss_to_exclude, ]

#write ss to exclude to file
write.csv(Ss_to_exclude, "Ss_to_exclude.csv")

df_concat$participant_id = factor(df_concat$participant_id)
df_concat$stakes = factor(df_concat$stakes)
df_concat$stakes
df_concat$stakes <- factor(df_concat$stakes, levels = rev(levels(df_concat$stakes)))
df_concat$stakes

#factor binary dependent variable for spineplot
df_concat$accurate_factor <- as.factor(df_concat$accurate)
df_concat$copied_factor <- as.factor(df_concat$copied)

table(df_concat$accurate_factor)/nrow(df_concat) #Overall accuracy throughout the experiment

# Spineplots of accuray by stakes
spineplot(accurate_factor ~ stakes, data = df_concat) # slightly higher accuracy at high stakes
spineplot(copied_factor ~ stakes, data = df_concat) # slightly higher copying at low stakes

# aggregate data for plotting
agg_df_concat <- ddply(.data = df_concat, .fun = summarise,
  .variables = c('participant_id', 'stakes'),# aggregate by 3 variables of interst 
  mean_copied = mean(copied),
  accuracy = mean(as.numeric(accurate)),
  mean_rt = mean(rt))
# Create plot of accuracy by stakes
ggplot(agg_df_concat, aes(x = stakes, y = accuracy, fill = stakes)) +
  geom_violinhalf(alpha = 0.3, color = NA) +
  geom_boxplot(width = 0.2, alpha = 0.5, size = 0.3) +
  geom_line(aes(group = participant_id), alpha = 0.1) + # Add lines connecting points
  stat_summary(fun = mean, geom = "point", size = 3, color = "red") +
  scale_fill_viridis_d(option = "plasma", direction = -1, begin = 0.05, end = 0.4) +
  labs(
    title = "Accuracy by Stakes and Payoff Structure",
    x = "Stakes Level", 
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    plot.title = element_text(size = 16)
  )
# Looks like participants are more accurate at high stakes
acc_model <- glmer(accurate ~ stakes + (1+stakes|participant_id),
                   family = binomial,
                   data = df_concat)
summary(acc_model)

#Print interpetable parameters
model_parameters(acc_model, digits = 3, exponentiate = TRUE)

#Right direction, but not significant
# Create effect plot for accuracy
plot_model(acc_model, 
          type = "pred", 
          terms = "stakes",
          title = "Predicted Probabilities of Accuracy by Stakes Level",
          axis.title = c("Stakes Level", "Predicted Probability of Accuracy")) +
  theme_minimal()


# Create plot of copying by stakes
ggplot(agg_df_concat, aes(x = stakes, y = mean_copied, fill = stakes)) +
  geom_violinhalf(alpha = 0.3, color = NA) +
  geom_boxplot(width = 0.2, alpha = 0.5, size = 0.3) +
  geom_line(aes(group = participant_id), alpha = 0.1) + # Add lines connecting points
#  geom_point(position = position_jitter(width = 0.01), alpha = 0.4, size = 2) +
  stat_summary(fun = mean, geom = "point", size = 3, color = "red") +
  scale_fill_viridis_d(option = "plasma", direction = -1, begin = 0.05, end = 0.4) +
  labs(
    title = "Copying by Stakes and Payoff Structure",
    x = "Stakes Level",
    y = "Copying",
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    plot.title = element_text(size = 16)
  )
# Looks like participants are more accurate at high stakes
copy_model <- glmer(copied ~ stakes + (1+stakes|participant_id),
                   family = binomial,
                   data = df_concat)
summary(copy_model)
#Pring interpetable parameters
model_parameters(copy_model, digits = 3, exponentiate = TRUE)

# Create effect plot for copying
plot_model(copy_model, 
          type = "pred", 
          terms = "stakes",
          title = "Predicted Probabilities of Copying by Stakes Level",
          axis.title = c("Stakes Level", "Predicted Probability of Copying")) +
  theme_minimal()

# Pre-registered analysis: critical trials
df_critical = df_concat[df_concat$critical == 1,] #filter for critical trials
unique(df_critical$day)
table(df_critical$day)
length(unique(df_critical$participant_id))
length(df_critical$participant_id)
#Spineplot of accuracy in critical trials
spineplot(accurate_factor ~ stakes, data = df_critical)
#Spineplot of copying in critical trials
spineplot(copied_factor ~ stakes, data = df_critical)
df_critical$accurate
# Aggregate data for plotting
agg_df_critical <- ddply(.data = df_critical, .fun = summarise,
  .variables = c('participant_id', 'stakes'),# aggregate by 3 variables of interst 
  mean_copied = mean(copied),
  accuracy = mean(accurate),
  mean_rt = mean(rt))
agg_df_critical$accuracy
# Create box plot of accuracy by stakes
# Create jitter positions that will be consistent between points and lines
set.seed(123)
jitter_pos <- position_jitter(width = 0.02, height = 0.01)

ggplot(agg_df_critical, aes(x = stakes, y = accuracy, fill = stakes)) +
  geom_boxplot(width = 0.2, alpha = 0.5) +
  geom_line(aes(group = participant_id), position = jitter_pos, alpha = 0.1, size = 0.5) + # Add lines connecting jittered points
  #geom_point(position = jitter_pos, alpha = 0.4, size = 2) +
  stat_summary(fun = mean, geom = "point", size = 3, color = "red") +
  scale_fill_viridis_d(option = "plasma", direction = -1, begin = 0.1, end = 0.45) +
  labs(
    title = "Proportion of Emulation-based Decisions in Critical Trials",
    x = "Stakes Level", 
    y = "Proportion of Emulation-based Decisions"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 16),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )
  
length(unique(df_critical$participant_id))
#model accuracy in critical trials
acc_critical_model <- glmer(accurate ~ stakes + (1+stakes|participant_id),
                   family = binomial,
                   data = df_critical)
summary(acc_critical_model)
#Print interpetable parameters
model_parameters(acc_critical_model, digits = 3, exponentiate = TRUE)

# Create effect plot for accuracy in critical trials
plot_model(acc_critical_model, 
          type = "pred", 
          terms = "stakes",
          title = "Predicted Probabilities of Accuracy in Critical Trials by Stakes Level",
          axis.title = c("Stakes Level", "Predicted Probability of Accuracy")) +
  theme_minimal()



# Read model posterior files
model1_posteriors <- read.csv("output/model1_posteriors.csv")
model2_posteriors <- read.csv("output/model2_posteriors.csv")
model3_posteriors <- read.csv("output/model3_posteriors.csv")

# Create a dataframe to compare beta values across models
beta_comparison <- data.frame(
  participant_id = model1_posteriors$participant_id,
  model1_beta = model1_posteriors$beta,
  model2_beta = model2_posteriors$beta,
  model3_beta = model3_posteriors$beta
)

# Determine which model has the highest beta for each participant
beta_comparison$highest_beta <- apply(beta_comparison[, c("model1_beta", "model2_beta", "model3_beta")], 1, 
                                     function(x) which.max(x))

# Convert to model names
beta_comparison$highest_beta_model <- factor(beta_comparison$highest_beta,
                                            levels = 1:3,
                                            labels = c("Model 1", "Model 2", "Model 3"))

# Calculate percentages
model_counts <- table(beta_comparison$highest_beta_model)
model_percentages <- prop.table(model_counts) * 100

# Print results
cat("Percentage of participants with highest beta by model:\n")
for (i in 1:length(model_percentages)) {
  cat(names(model_percentages)[i], ": ", round(model_percentages[i], 2), "%\n", sep="")
}

# Create a bar plot of the percentages
ggplot(data.frame(Model = names(model_percentages), 
                  Percentage = as.numeric(model_percentages)), 
       aes(x = Model, y = Percentage, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = paste0(round(Percentage, 1), "%")), 
            vjust = -0.5, size = 4) +
  labs(title = "Percentage of Participants with Highest Beta by Model",
       y = "Percentage of Participants") +
  theme_minimal() +
  theme(legend.position = "none")

