# ---------------- LOAD PACKAGES -------------

library(readxl)
library(gt)
library(dplyr)
library(ggplot2)
library(tidyr)

# ---------------- LOAD DATA -----------------

data <- read_excel("T:/Research_01/CZE-2020.67 - SAVI-AoS/output_danny/data_stenosis_complete.xlsx")
head(data)

# PREPROCESSING
# Removing where calcification was not found, these are patients without segmentations
removed_rows <- data[is.na(data$calc_volume), ]
data   <- data[!is.na(data$calc_volume), ]

n_removed <- nrow(removed_rows)
n_total   <- nrow(data)

cat("Removed patients:", n_removed, "\n")
cat("Remaining patients:", nrow(data), "\n") # 8 patients were removed

# When calcificaiton was not found, change NA to 0 
calc_cols <- grep("calc", colnames(data), value = TRUE)
data[calc_cols][is.na(data[calc_cols])] <- 0

data_male <- data %>% filter(Gender_Score == 1)
data_female <- data %>% filter(Gender_Score == 0)

data$total_central_calc <- data$central_NCC_calc +
  data$central_LCC_calc +
  data$central_RCC_calc

data$total_peripheral_calc <- data$peripheral_NCC_calc +
  data$peripheral_LCC_calc +
  data$peripheral_RCC_calc



# Remove spaces, commas, or other non-numeric characters and convert to numeric
data <- data %>%
  mutate(
    `peak_velocity` = as.numeric(gsub(",", ".", `Avmaxpre (cm/s)`)),
    AVApre = as.numeric(gsub(",", ".", AVApre)),
    meanRR = as.numeric(gsub(",", ".", meanRR)),
    SAVI = as.numeric(gsub(",", ".", SAVI))
  )


# ---------------- DEMOGRAPHICS ----------------

# Sample size
n <- nrow(data)

# Create table
demographics_table <- data.frame(
  Variable = c(
    "Number of patients",
    "Age (years)",
    "Male (%)",
    "Length (cm)",
    "Weight (kg)"
  ),
  Value = c(
    n,
    sprintf("%.1f ± %.1f", mean(data$Age, na.rm = TRUE), sd(data$Age, na.rm = TRUE)),
    sprintf("%.1f %%", mean(data$Gender_Score, na.rm = TRUE) * 100),
    sprintf("%.1f ± %.1f", mean(data$Length, na.rm = TRUE), sd(data$Length, na.rm = TRUE)),
    sprintf("%.1f ± %.1f", mean(data$Weight, na.rm = TRUE), sd(data$Weight, na.rm = TRUE))
  )
)





# ---------------- CREATE DEMOGRAPHICS GT TABLE --------------

table1 <- demographics_table %>%
  gt() %>%
  tab_header(
    title = "Table 1",
    subtitle = "Patient Demographics"
  ) %>%
  cols_label(
    Variable = "Variable",
    Value = "Value"
  ) %>%
  cols_align(
    align = "left",
    columns = Variable
  ) %>%
  cols_align(
    align = "center",
    columns = Value
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_options(
    table.border.top.width = px(2),
    table.border.bottom.width = px(2)
  ) %>%
  tab_footnote(
    footnote = "Values are mean ± standard deviation; Gender is reported as percentage of males.",
    locations = cells_title(groups = "subtitle")
  )

table1




# ---------------- CREATE DEMOGRAPHICS GT TABLE SPLIT BY GENDER --------------------

data_male <- data %>% filter(Gender_Score == 1)
data_female <- data %>% filter(Gender_Score == 0)

library(gt)

# Function to create a compact table
create_g <- function(df, title, footnote_text = NULL) {
  gt(df) %>%
    tab_header(title = title) %>%
    cols_label(
      Variable = "Variable",
      Value = "Value"
    ) %>%
    cols_align(
      align = "left",
      columns = Variable
    ) %>%
    cols_align(
      align = "center",
      columns = Value
    ) %>%
    tab_style(
      style = cell_text(weight = "bold"),
      locations = cells_column_labels()
    ) %>%
    tab_options(
      table.border.top.width = px(0),    # IEEE prefers minimal borders
      table.border.bottom.width = px(0),
      table.font.size = px(10),
      data_row.padding = px(2),
      heading.padding = px(2)
    ) %>%
    {if (!is.null(footnote_text)) tab_footnote(., footnote = footnote_text, locations = cells_title(groups = "title")) else .}
}

table1

# Full path
writeLines(as_latex(table1),
           "H:/DATA/Afstuderen/3.Data/figures/Table1_Demographics.tex")


# ----------------- EXPLORATORY DATA ANALYSIS ----------------
#CHECK DISTRIBUTIONS OF DATA

shapiro.test(data$SAVI) # Normally distributed
hist(data$SAVI, breaks =8)
plot(cut(data$SAVI, breaks = 4))

shapiro.test(data$AVApre) # Not normally distributed
hist(data$AVApre) # Right skewed, large AVAs
shapiro.test(data$`peak_velocity`) #Normally distributed
shapiro.test(data$meanRR) # Normally distributed

shapiro.test(data$calc_volume) # Not normally distributed
hist(data$calc_volume)

shapiro.test(data$LCC_calc) # Not normally distributed
hist(data$LCC_calc) # Right skewed 

shapiro.test(data$NCC_calc) # Not normally distributed
hist(data$NCC_calc) # Right skewed

shapiro.test(data$RCC_calc) # not normally distributed
hist(data$RCC_calc) # Right skewed

shapiro.test(data$calc_volume[data$Gender_Score==0])  # Female, normally distributed
shapiro.test(data$calc_volume[data$Gender_Score==1])  # Male, not normally distributed
hist(data$calc_volume[data$Gender_Score==1]) # Male, right skewed
var.test(calc_volume ~ Gender, data = data) # Variances are equal

#CHECK COLLINEARITY
# Select variables of interest
vars <- data[, c("SAVI", "AVApre", "peak_velocity", "meanRR")]
cor(vars, use = "complete.obs", method = "pearson")

# Define a custom panel function for linear regression
panel.lm <- function(x, y, ...) {
  points(x, y, ...)            # plot the points
  abline(lm(y ~ x), col = "red")  # add linear regression line
}

# Now create the scatterplot matrix
pairs(data[, c("SAVI", "AVApre", "peak_velocity", "meanRR")],
      panel = panel.lm,
      main = "Scatterplot Matrix with Linear Fits")

# No large correlations between stenosis severity parameters



# ---------------- STENOSIS SEVERITY PARAMETERS TABLE ----------------


clinical_table <- data.frame(
  Variable = c(
    "peak_velocity",
    "AVApre",
    "meanRR (mmHg)",
    "SAVI"
  ),
  Value = c(
    sprintf("%.1f ± %.1f", mean(data$peak_velocity, na.rm = TRUE), sd(data$peak_velocity, na.rm = TRUE)),
    sprintf("%.2f ± %.2f", mean(data$AVApre, na.rm = TRUE), sd(data$AVApre, na.rm = TRUE)),
    sprintf("%.1f ± %.1f", mean(data$meanRR, na.rm = TRUE), sd(data$meanRR, na.rm = TRUE)),
    sprintf("%.2f ± %.2f", mean(data$SAVI, na.rm = TRUE), sd(data$SAVI, na.rm = TRUE))
  )
)

table_clinical <- clinical_table %>%
  gt() %>%
  tab_header(
    title = "Table 2: Clinical Variables"
  ) %>%
  cols_label(
    Variable = "Variable",
    Value = "Value"
  ) %>%
  cols_align(
    align = "left",
    columns = Variable
  ) %>%
  cols_align(
    align = "center",
    columns = Value
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_options(
    table.border.top.width = px(0),    # compact IEEE style
    table.border.bottom.width = px(0),
    table.font.size = px(10),
    data_row.padding = px(2),
    heading.padding = px(2)
  ) %>%
  tab_footnote(
    footnote = "Values are mean ± SD.",
    locations = cells_title(groups = "title")
  )

#  SAVE TABLE AS LATEX
writeLines(as_latex(table_clinical),
           "H:/DATA/Afstuderen/3.Data/figures/Table2_Clinical.tex")

table_clinical




# ---------------- CLEAN DATA ----------------
data <- data %>%
  mutate(
    `Avmaxpre (cm/s)` = as.numeric(`Avmaxpre (cm/s)`),
    AVApre = as.numeric(AVApre),
    meanRR = as.numeric(meanRR),
    SAVI = as.numeric(SAVI),
    Gender = factor(Gender_Score, labels = c("Female", "Male"))
  )

# Function to create 3-box plot per parameter
create_three_boxplot <- function(df, parameter_name, colors, file_name) {
  
  # Prepare data: All / Female / Male
  param_combined <- df %>%
    select(all_of(parameter_name), Gender) %>%
    rename(Value = all_of(parameter_name)) %>%
    mutate(Group = "All patients") %>%
    bind_rows(
      df %>% filter(Gender == "Female") %>% select(all_of(parameter_name), Gender) %>% 
        rename(Value = all_of(parameter_name)) %>% mutate(Group = "Female"),
      df %>% filter(Gender == "Male") %>% select(all_of(parameter_name), Gender) %>% 
        rename(Value = all_of(parameter_name)) %>% mutate(Group = "Male")
    )
  
  # Plot
  p <- ggplot(param_combined, aes(x = Group, y = Value, fill = Group)) +
    geom_boxplot(outlier.shape = 21, color = "black", alpha = 0.7) +
    geom_jitter(width = 0.1, size = 1.5, alpha = 0.6, color = "black") +
    theme_minimal(base_size = 12) +
    labs(
      title = paste0(parameter_name, " Distribution by Group"),
      x = "",
      y = parameter_name
    ) +
    scale_fill_manual(values = colors) +
    theme(legend.position = "none")
  
  # Save
  ggsave(filename = file_name, plot = p, width = 7, height = 5, dpi = 300)
  
  return(p)
}

# ---------------- CREATE FIGURES ----------------
colors <- c("lightblue", "pink", "skyblue")

# Avmaxpre
plot_avmax <- create_three_boxplot(data, "peak_velocity", colors,
                                   "H:/DATA/Afstuderen/3.Data/figures/Avmaxpre_AllGender.png")

# AVApre
plot_ava <- create_three_boxplot(data, "AVApre", colors,
                                 "H:/DATA/Afstuderen/3.Data/figures/AVApre_AllGender.png")

# meanRR
plot_meanRR <- create_three_boxplot(data, "meanRR", colors,
                                    "H:/DATA/Afstuderen/3.Data/figures/meanRR_AllGender.png")

# SAVI
plot_savi <- create_three_boxplot(data, "SAVI", colors,
                                  "H:/DATA/Afstuderen/3.Data/figures/SAVI_AllGender.png")

# Test whether there is a significant difference in peak_velocity by gender
plot_avmax
shapiro.test(data$peak_velocity[data$Gender_Score==0])  # Female, normally distributed
shapiro.test(data$peak_velocity[data$Gender_Score==1])  # Male, normally distributed
t_test_avmax_gender <- t.test(peak_velocity ~ Gender, data = data, var.equal = TRUE)
t_test_avmax_gender # No significant difference


# Test whether there is a significant difference in AVA by gender
plot_ava
shapiro.test(data$AVApre[data$Gender_Score==0])  # Female, normally distributed
shapiro.test(data$AVApre[data$Gender_Score==1])  # Male, normally distributed
t_test_ava_gender <- t.test(AVApre ~ Gender, data = data, var.equal = TRUE)
t_test_ava_gender # Statist ical difference. Male AVA is bigger.

# Test whether there is a significant difference in meanRR by gender
plot_meanRR
shapiro.test(data$meanRR[data$Gender_Score==0])  # Female, not normally distributed
shapiro.test(data$meanRR[data$Gender_Score==1])  # Male, normally distributed
# Wilcoxon rank-sum test
wilcox_test_meanRR_gender <- wilcox.test(meanRR ~ Gender, data = data)
wilcox_test_meanRR_gender # No difference

# Test whether there is a significant difference in SAVI by gender
plot_savi
shapiro.test(data$SAVI[data$Gender_Score==0])  # Female, normally distributed
shapiro.test(data$SAVI[data$Gender_Score==1])  # Male, normally distributed
t_test_savi_gender <- t.test(SAVI ~ Gender, data = data, var.equal = TRUE)
t_test_savi_gender # No difference

# ------------------------- CALC VOLUME BETWEEN GENDERS -------------------------

# Ensure Gender is a factor with proper labels
data$Gender <- factor(data$Gender_Score, labels = c("Female", "Male"))

# Run standard t-test
t_test_calc_gender <- t.test(calc_volume ~ Gender, data = data, var.equal = TRUE)

# View results
t_test_calc_gender # No Statistical difference 

ggplot(data, aes(x = Gender, y = calc_volume, fill = Gender)) +
  geom_boxplot(outlier.shape = 21, color = "black", alpha = 0.7) +
  geom_jitter(width = 0.1, size = 1.5, alpha = 0.6, color = "black") +
  labs(
    title = "Aortic Valve Calcification Volume by Gender",
    x = "Gender",
    y = "Calcification Volume (mm³)"
  ) +
  scale_fill_manual(values = c("pink", "skyblue")) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")



# ------------------------------ OVERALL CALC VOLUME CORRELATIONS ----------------------------

# First test correlations with the overal calcification - spearman, as not normally distributed
cor.test(data$calc_volume, data$AVApre, method = "spearman")
# There is no significantl ienar ssociation between aortic valve calcification volume and AVA in this cohor

cor.test(data$calc_volume, data$SAVI, method = "pearson") # SIGNIFICANT RELATIONSHIP
cor_result <- cor.test(data$calc_volume, data$SAVI, method = "pearson")

plot(data$calc_volume, data$SAVI,
     xlab = "Calcification Volume",
     ylab = "SAVI",
     main = "Calcification Volume vs SAVI",
     pch = 19)

abline(lm(SAVI ~ calc_volume, data = data), lwd = 2)

legend("topright",
       legend = paste0("r = ", round(cor_result$estimate, 2),
                       "\np = ", signif(cor_result$p.value, 3)),
       bty = "n")
# Significant inverse relationship. More calcification, lower SAVI

cor.test(data$calc_volume, data$peak_velocity, method = "pearson")
# Non significant relationship.

cor.test(data$calc_volume, data$meanRR, method = "pearson")
# Non significant




# ----------------------------------------------- LEAFLET SPECIFIC CORRELATION ------------------------

data <- data %>%
  mutate(
    SAVI = as.numeric(SAVI),
    LCC_calc = as.numeric(LCC_calc),
    NCC_calc = as.numeric(NCC_calc),
    RCC_calc = as.numeric(RCC_calc)
  )

# Test whether normality of distribution is true
shapiro.test(data$LCC_calc) # Not normally distributed, p< 0.001
shapiro.test(data$NCC_calc) # Not normally distributed, p = 0.03
shapiro.test(data$RCC_calc) # Not normally distributed, p = 0.03

friedman.test(
  y = as.matrix(data[, c("NCC_calc", "LCC_calc", "RCC_calc")])
)# Significant difference!

# Columns to compare
cols <- c("NCC_calc", "LCC_calc", "RCC_calc")

# Generate all pairwise combinations
pairs <- combn(cols, 2, simplify = FALSE)

# Run paired Wilcoxon test for each pair and get medians
results <- lapply(pairs, function(p) {
  median1 <- median(data[[p[1]]], na.rm = TRUE)
  median2 <- median(data[[p[2]]], na.rm = TRUE)
  test <- wilcox.test(data[[p[1]]], data[[p[2]]], paired = TRUE, exact = FALSE)
  direction <- ifelse(median1 > median2, paste(p[1], ">", p[2]),
                      ifelse(median1 < median2, paste(p[2], ">", p[1]), "equal"))
  c(pair = paste(p[1], "vs", p[2]),
    median1 = median1,
    median2 = median2,
    direction = direction,
    p_value = test$p.value)
})

# Convert to data frame
results_df <- do.call(rbind, results)
results_df <- as.data.frame(results_df)
results_df$median1 <- as.numeric(as.character(results_df$median1))
results_df$median2 <- as.numeric(as.character(results_df$median2))
results_df$p_value <- as.numeric(as.character(results_df$p_value))

# Apply Bonferroni correction
results_df$p_adjusted <- p.adjust(results_df$p_value, method = "bonferroni")

results_df
####
# SAVI
###
leaflet_long <- data %>%
  select(LCC_calc, NCC_calc, RCC_calc, SAVI) %>%
  pivot_longer(
    cols = c(LCC_calc, NCC_calc, RCC_calc),
    names_to = "Leaflet",
    values_to = "CalcVolume"
  )

# BOXPLOTS 
leaflet_plot <- ggplot(leaflet_long, aes(x = Leaflet, y = CalcVolume, fill = Leaflet)) +
  geom_boxplot(outlier.shape = 21, color = "black", alpha = 0.7) +
  geom_jitter(aes(color = Leaflet),
              width = 0.15,
              size = 1.5,
              alpha = 0.6,
              show.legend = FALSE) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Leaflet-specific Calcification Distribution",
    x = "Leaflet",
    y = "Calcification Volume (mm³)"
  ) +
  scale_fill_manual(values = c("pink", "lightblue", "skyblue")) +
  theme(legend.position = "none")

# SHOW PLOT
print(leaflet_plot)

# Test correlations with leaflet-specific calcification 
cor.test(data$LCC_calc, data$SAVI, method = "spearman") # Non-significant
cor.test(data$NCC_calc, data$SAVI, method = "spearman") # Significant relationship, p = 0.004
cor.test(data$RCC_calc, data$SAVI, method = "spearman") # Non-significant

# Scatterplot between SAVI and NCC, as this is significant
ggplot(data, aes(x = NCC_calc, y = SAVI)) +
  geom_point(size = 3, color = "steelblue") +          # scatter points
  geom_smooth(method = "lm", color = "firebrick",      # regression line
              se = TRUE, linetype = "solid") +        # 95% CI shaded
  labs(
    title = "Relationship between NCC Calcification and SAVI",
    x = "NCC Calcification [mm^3]",
    y = "SAVI"
  ) +
  theme_minimal(base_size = 14)

# REGRESSION MODEL
savi_regression = lm(SAVI ~ NCC_calc + LCC_calc + RCC_calc, data = data) # NCC calc is significant
summary(savi_regression)
par(mfrow = c(2,2))
plot(savi_regression)

library(car)
vif(lm(SAVI ~ NCC_calc + RCC_calc + LCC_calc, data = data))

##
## AVA
### Test correlations with leaflet-specific calcification 
cor.test(data$LCC_calc, data$AVApre, method = "spearman") # Non-significant
cor.test(data$NCC_calc, data$AVApre, method = "spearman") # Non significant
cor.test(data$RCC_calc, data$AVApre, method = "spearman") # Non-significant

ava_regression = lm(AVApre ~ NCC_calc + LCC_calc + RCC_calc, data = data) # SIGNIFICANT
summary(ava_regression)
par(mfrow = c(2,2))
plot(ava_regression)

# Test for multicollinearity
vif(lm(AVApre ~ NCC_calc + RCC_calc + LCC_calc, data = data))


###
# PEAK VELOCITY 

# Test correlations with leaflet-specific calcification 
cor.test(data$LCC_calc, data$peak_velocity, method = "spearman") # Non significant
cor.test(data$NCC_calc, data$peak_velocity, method = "pearson") # Non significant
cor.test(data$RCC_calc, data$peak_velocity, method = "spearman") # Significant! 


velocity_regression = lm(peak_velocity ~ NCC_calc + LCC_calc + RCC_calc, data = data) # Non significant
summary(velocity_regression)
par(mfrow = c(2,2))
plot(velocity_regression)



####
#  MEAN GRADIENT
# Test correlations with leaflet-specific calcification 
cor.test(data$LCC_calc, data$meanRR, method = "spearman") # Non significant
cor.test(data$NCC_calc, data$meanRR, method = "pearson") # Non significant
cor.test(data$RCC_calc, data$meanRR, method = "spearman") # Non significant

meanrr_regression = lm(meanRR ~ NCC_calc + LCC_calc + RCC_calc, data = data) # Non significant
summary(meanrr_regression)
par(mfrow = c(2,2))
plot(meanrr_regression)

# ----------------------------------------------- CENTRAL / PERIPHERAL CORRELATION ------------------------

shapiro.test(data$total_central_calc) # Not normally distributed, p = 0.04
hist(data$total_central_calc)
shapiro.test(data$total_peripheral_calc) # Not normally distributed, p=0.02
hist(data$total_peripheral_calc)

# Gather the two variables into long format
data_long <- data %>%
  pivot_longer(
    cols = c(total_central_calc, total_peripheral_calc),
    names_to = "Location",
    values_to = "Calcification"
  )

# Plot side-by-side boxplots
ggplot(data_long, aes(x = Location, y = Calcification, fill = Location)) +
  geom_boxplot(width = 0.6, outlier.color = "red", outlier.shape = 16) +
  scale_fill_manual(values = c("steelblue", "firebrick")) +
  labs(
    title = "Total Central vs Peripheral Calcification",
    x = "Location",
    y = "Calcification Volume"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

wilcox.test(data$total_peripheral_calc,
            data$total_central_calc,
            paired = FALSE,           # unpaired
            alternative = "two.sided", # test for difference in either direction
            na.action = na.omit)       # ignore missing values

####
# SAVI 

# Test correlations with central-peripheral calcification 
cor.test(data$total_central_calc, data$SAVI, method = "spearman") # Significant relationship!!
cor.test(data$total_peripheral_calc, data$SAVI, method = "spearman")  #Non-significant 

savi_regression_central = lm(SAVI ~ total_peripheral_calc + total_central_calc, data = data) # Non significant
summary(savi_regression_central)
par(mfrow = c(2,2))
plot(savi_regression_central)



####
# AVA

# Test correlations with central-peripheral calcification 
cor.test(data$total_central_calc, data$AVApre, method = "spearman") # Non-significant
cor.test(data$total_peripheral_calc, data$AVApre, method = "spearman")  #Non-significant 

ava_regression_central = lm(AVApre ~ total_peripheral_calc + total_central_calc, data = data) # Non significant
summary(ava_regression_central)
par(mfrow = c(2,2))
plot(ava_regression_central)


###
#  Mean Gradient 

# Test correlations with central-peripheral calcification 
cor.test(data$total_central_calc, data$meanRR, method = "spearman") # Non-significant
cor.test(data$total_peripheral_calc, data$meanRR, method = "spearman")  #Non-significant 

meanrr_regression_central = lm(meanRR ~ total_peripheral_calc + total_central_calc, data = data) # Non significant
summary(meanrr_regression_central)
par(mfrow = c(2,2))
plot(meanrr_regression_central)


###
# Velocity 

# Test correlations with central-peripheral calcification 
cor.test(data$total_central_calc, data$peak_velocity, method = "spearman") # Non-significant
cor.test(data$total_peripheral_calc, data$peak_velocity, method = "spearman")  #Non-significant 

peak_velocity_regression_central = lm(peak_velocity ~ total_peripheral_calc + total_central_calc, data = data) # Non significant
summary(peak_velocity_regression_central)
par(mfrow = c(2,2))
plot(peak_velocity_regression_central)


# ----------------------------------------------- WITHIN LEAFLET ANALYSIS ---------------------------

# Test whether normality of distribution is true
shapiro.test(data$central_NCC_calc) # not normal
shapiro.test(data$peripheral_NCC_calc) # not normal

shapiro.test(data$central_RCC_calc) # not normal, = 0 .001
hist(data$central_RCC_calc)
shapiro.test(data$peripheral_RCC_calc) #not normal

shapiro.test(data$central_LCC_calc) # not normal
shapiro.test(data$peripheral_LCC_calc) # not normal, = 0.001
hist(data$peripheral_LCC_calc)


# Analzye whether correlation between regions exist
calc_vars <- data[, c(
  "central_NCC_calc", "peripheral_NCC_calc",
  "central_LCC_calc", "peripheral_LCC_calc",
  "central_RCC_calc", "peripheral_RCC_calc")]

cor_matrix <- cor(calc_vars, use = "complete.obs", method = "pearson") # High correlation between peripheral NCC and LCC calcification
cor_matrix

plot(data$peripheral_NCC_calc, data$peripheral_LCC_calc,
     xlab = "Peripheral NCC Calc",
     ylab = "Peripheral LCC Calc",
     main = "Scatter: Peripheral NCC vs LCC")
abline(lm(peripheral_LCC_calc ~ peripheral_NCC_calc, data = data), col = "red")

# LOOK AT HOW THE CALCIFICATION IS DISTRIBUTED
# 1. Specify the 6 regions
region_vars <- c(
  "central_NCC_calc", "peripheral_NCC_calc",
  "central_LCC_calc", "peripheral_LCC_calc",
  "central_RCC_calc", "peripheral_RCC_calc"
)

# 2. Compute sum of 6 regions per patient
data$sum_regions <- rowSums(data[, region_vars], na.rm = TRUE)

# 3. Compute % contribution per patient
data_prop <- data
data_prop[, region_vars] <- data[, region_vars] / data$sum_regions * 100  # now in percent

# 4. Compute mean and SD across patients for each region
mean_props <- apply(data_prop[, region_vars], 2, mean, na.rm = TRUE)
sd_props   <- apply(data_prop[, region_vars], 2, sd, na.rm = TRUE)

# 5. Put it in a nice table
mean_sd_table <- data.frame(
  Region = region_vars,
  Mean_Percentage = mean_props,
  SD_Percentage = sd_props
)

# Sort by mean contribution descending (optional)
mean_sd_table <- mean_sd_table[order(-mean_sd_table$Mean_Percentage), ]
mean_sd_table


###
## 1. SAVI
###

# Define your 6 calcification regions
region_vars <- c(
  "central_NCC_calc", "peripheral_NCC_calc",
  "central_LCC_calc", "peripheral_LCC_calc",
  "central_RCC_calc", "peripheral_RCC_calc"
)

# Define the outcome variables
outcomes <- c("SAVI", "meanRR", "peak_velocity", "AVApre")

# Initialize an empty list to store results
cor_results_list <- list()

# Loop over each outcome and region
for (outcome in outcomes) {
  cor_results <- lapply(region_vars, function(var) {
    cor_test <- cor.test(data[[var]], data[[outcome]], method = "spearman", use = "complete.obs")
    data.frame(
      Outcome = outcome,
      Region = var,
      rho = cor_test$estimate,
      p_value = cor_test$p.value
    )
  })
  
  # Combine and add to the main list
  cor_results_list[[outcome]] <- do.call(rbind, cor_results)
}

# Combine all outcomes into one data frame
cor_results_df <- do.call(rbind, cor_results_list)

# View results
cor_results_df

savi_lm_total = lm(SAVI ~  peripheral_NCC_calc + peripheral_LCC_calc + peripheral_RCC_calc +
                     central_NCC_calc + central_LCC_calc + central_RCC_calc, data = data) # Non significant
summary(savi_lm_total)


###
## 1. AVA
###

# Correlatioanl tests with SAVI
results_ava <- lapply(region_vars, function(var) {
  cor_test <- cor.test(data[[var]], data$AVApre, method = "spearman", use = "complete.obs")
  data.frame(
    Region = var,
    rho = cor_test$estimate,
    p_value = cor_test$p.value
  )
})

# Combine into a single data frame
results_ava_df <- do.call(rbind, results_ava)
results_ava_df

# Linear model for SAVI
savi_lm_total = lm(SAVI ~  peripheral_NCC_calc + peripheral_LCC_calc + peripheral_RCC_calc +
                     central_NCC_calc + central_LCC_calc + central_RCC_calc, data = data) # Non significant
summary(savi_lm_total)


# Model for peak_velocity
peak_velocity_lm_total <- lm(peak_velocity ~ peripheral_NCC_calc + peripheral_LCC_calc + peripheral_RCC_calc +
                               central_NCC_calc + central_LCC_calc + central_RCC_calc, data = data)
summary(peak_velocity_lm_total)


# Model for meanRR
meanRR_lm_total <- lm(meanRR ~ peripheral_NCC_calc + peripheral_LCC_calc + peripheral_RCC_calc +
                        central_NCC_calc + central_LCC_calc + central_RCC_calc, data = data)
summary(meanRR_lm_total)

# Model for AVApre
AVApre_lm_total <- lm(AVApre ~ peripheral_NCC_calc + peripheral_LCC_calc + peripheral_RCC_calc +
                        central_NCC_calc + central_LCC_calc + central_RCC_calc, data = data)
summary(AVApre_lm_total)

