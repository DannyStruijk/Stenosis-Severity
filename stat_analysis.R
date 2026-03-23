# ---------------- LOAD PACKAGES -------------

library(readxl)
library(gt)
library(dplyr)
library(ggplot2)
library(tidyr)

# ---------------- LOAD DATA -----------------

data <- read_excel("T:/Research_01/CZE-2020.67 - SAVI-AoS/output_danny/data_stenosis_severity.xlsx")
head(data)

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

names(data)[names(data) == "Avmaxpre (cm/s)"] <- "peak_velocity"

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
shapiro.test(data$AVApre) # Not normally distributed
hist(data$AVApre) # Right skewed, large AVAs
shapiro.test(data$`peak_velocity`) #Normally distributed
shapiro.test(data$meanRR) # Normally distributed

shapiro.test(data$calc_volume) # Normally distributed
shapiro.test(data$LCC_calc) # Not normally distributed
shapiro.test(data$NCC_calc) # Normally distributed
shapiro.test(data$RCC_calc) # not normally distributed

shapiro.test(data$calc_volume[data$Gender_Score==0])  # Female, normally distributed
shapiro.test(data$calc_volume[data$Gender_Score==1])  # Male, normally distributed
var.test(calc_volume ~ Gender, data = data) # Variances are equal

#CHECK COLLINEARITY
# Select variables of interest
vars <- data[, c("SAVI", "AVApre", "peak_velocity", "meanRR")]
cor(vars, use = "complete.obs", method = "pearson")

pairs(data[, c("SAVI", "AVApre", "peak_velocity", "meanRR")], 
      panel = panel.smooth,
      main = "Scatterplot Matrix")


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



# BOXPLOTS  SAVI
library(dplyr)
library(ggplot2)

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

plot_avmax
plot_ava
plot_meanRR
plot_savi

# ------------------------- CALC VOLUME BETWEEN GENDERS -------------------------

# Ensure Gender is a factor with proper labels
data$Gender <- factor(data$Gender_Score, labels = c("Female", "Male"))

# Run standard t-test
t_test_calc_gender <- t.test(calc_volume ~ Gender, data = data, var.equal = TRUE)

# View results
t_test_calc_gender

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

cor.test(data$calc_volume, data$SAVI, method = "pearson")
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
shapiro.test(data$NCC_calc)
shapiro.test(data$RCC_calc) # Not normally distributed, p = 0.03



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
cor.test(data$NCC_calc, data$SAVI, method = "pearson") # Significant relationship, p = 0.01
cor.test(data$RCC_calc, data$SAVI, method = "pearman") # Non-significant

# Scatterplot between SAVI and NCC, as this is significant
ggplot(data, aes(x = NCC_calc, y = SAVI)) +
  geom_point(size = 3, color = "steelblue") +          # scatter points
  geom_smooth(method = "lm", color = "firebrick",      # regression line
              se = TRUE, linetype = "solid") +        # 95% CI shaded
  labs(
    title = "Relationship between NCC Calcification and SAVI",
    x = "NCC Calcification",
    y = "SAVI"
  ) +
  theme_minimal(base_size = 14)

# REGRESSION MODEL
savi_regression = lm(SAVI ~ NCC_calc + LCC_calc + RCC_calc, data = data) # NOTHING IS SIGNIFICANT
summary(savi_regression)
par(mfrow = c(2,2))
plot(savi_regression)



##
## AVA
### Test correlations with leaflet-specific calcification 
cor.test(data$LCC_calc, data$AVApre, method = "spearman") # Non-significant
cor.test(data$NCC_calc, data$AVApre, method = "pearson") # Non significant
cor.test(data$RCC_calc, data$AVApre, method = "spearman") # Non-significant

ava_regression = lm(AVApre ~ NCC_calc + LCC_calc + RCC_calc, data = data) # SIGNIFICANT
summary(ava_regression)
par(mfrow = c(2,2))
plot(ava_regression)


###
# PEAK VELOCITY 

# Test correlations with leaflet-specific calcification 
cor.test(data$LCC_calc, data$peak_velocity, method = "spearman") # Non significant
cor.test(data$NCC_calc, data$peak_velocity, method = "pearson") # Non significant
cor.test(data$RCC_calc, data$peak_velocity, method = "spearman") # Non significant

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
shapiro.test(data$total_peripheral_calc) # Not normally distributed, p=0.02

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


####
# SAVI 

# Test correlations with central-peripheral calcification 
cor.test(data$total_central_calc, data$SAVI, method = "spearman") # Non-significant
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
shapiro.test(data$central_NCC_calc) # normal
shapiro.test(data$peripheral_NCC_calc) # normal

shapiro.test(data$central_RCC_calc) # not normal, = 0 .001
shapiro.test(data$peripheral_RCC_calc) #normal

shapiro.test(data$central_LCC_calc) # normal
shapiro.test(data$peripheral_LCC_calc) # not normal, = 0.001

calc_vars <- data[, c(
  "central_NCC_calc", "peripheral_NCC_calc",
  "central_LCC_calc", "peripheral_LCC_calc",
  "central_RCC_calc", "peripheral_RCC_calc"
)]

cor_matrix <- cor(calc_vars, use = "complete.obs", method = "pearson") # High correlation between peripheral NCC and LCC calcification
cor_matrix



