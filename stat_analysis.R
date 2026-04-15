# ---------------- LOAD PACKAGES -------------

library(readxl)
library(gt)
library(dplyr)
library(ggplot2)
library(tidyr)

# ---------------- LOAD DATA -----------------

data <- read_excel("T:/Research_01/CZE-2020.67 - SAVI-AoS/output_danny/patient_data_with_volumes_and_areas.xlsx")
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

# Define colors
color_lcc <- "#1f77b4"  # blue
color_rcc <- "#ff7f0e"  # orange
color_ncc <- "#2ca02c"  # green

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
    sprintf("%.1f ± %.1f", mean(data$weight, na.rm = TRUE), sd(data$weight, na.rm = TRUE))
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

shapiro.test(data$calc_volume) # normally distributed
hist(data$calc_volume)

shapiro.test(data$calc_volume_indexed) # normally distributed
hist(data$calc_volume_indexed)

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
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor = 1.2, ...) {
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  
  r <- cor(x, y, use = "complete.obs")
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  
  text(0.5, 0.5, txt, cex = cex.cor)
}

pairs(
  data[, c("SAVI", "AVApre", "peak_velocity", "meanRR")],
  lower.panel = panel.lm,
  upper.panel = panel.cor,
  main = "Scatterplot Matrix with Linear Fits and Correlations"
)

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


shapiro.test(data$calc_volume) # Normally distributed
shapiro.test(data$SAVI) # Normally distributed


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

# IEEE FORMAT TOTAL CALCIFICATION AGAINST SAVI
# Compute correlation and p-value
cor_result <- cor.test(data$calc_volume, data$SAVI, use = "complete.obs")
r <- cor_result$estimate
p <- cor_result$p.value

ggplot(data, aes(x = calc_volume, y = SAVI)) +
  geom_point(
    size = 2,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",
    linewidth = 0.8,
    linetype = "solid"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "r = ", round(r, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("Calcification Volume ("*mm^3*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )

# Indexed overall calclfication 

data$total_valve_area <- data$LCC_total_area_mm2 +
  data$NCC_total_area_mm2 +
  data$RCC_total_area_mm2

# Indexed calcification (density)
data$calc_volume_indexed <- data$calc_volume / data$total_valve_area
shapiro.test(data$calc_volume_indexed) # Normally distributed



cor_result <- cor.test(data$calc_volume_indexed, data$SAVI, use = "complete.obs")
r <- cor_result$estimate
p <- cor_result$p.value


ggplot(data, aes(x = calc_volume_indexed, y = SAVI)) +
  geom_point(
    size = 2,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",
    linewidth = 0.8,
    linetype = "solid"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "r = ", round(r, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("Aortic Valve Calcification ("*mm^3/mm^2*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )







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
shapiro.test(data$NCC_calc) # Normally distributed, p = 0.03
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
    y = "Aortic Valve Calcification (mm³)"
  ) +
  scale_fill_manual(values = c("pink", "lightblue", "skyblue")) +
  theme(legend.position = "none")

# SHOW PLOT
print(leaflet_plot)


 ### iEEE FORMAT 


leaflet_long <- data %>%
  select(LCC_calc, NCC_calc, RCC_calc, SAVI) %>%
  pivot_longer(
    cols = c(LCC_calc, NCC_calc, RCC_calc),
    names_to = "Leaflet",
    values_to = "CalcVolume"
  ) %>%
  mutate(
    Leaflet = factor(Leaflet, levels = c("NCC_calc", "LCC_calc", "RCC_calc"),
                     labels = c("NCC", "LCC", "RCC"))
  )

leaflet_plot <- ggplot(leaflet_long, aes(x = Leaflet, y = CalcVolume, fill = Leaflet)) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    color = "black",
    linewidth = 1
  ) +
  geom_jitter(
    aes(color = Leaflet),
    width = 0.12,
    size = 2,
    alpha = 0.7,
    show.legend = FALSE
  ) +
  scale_fill_manual(values = c(
    "LCC" = "#1f77b4",
    "RCC" = "#ff7f0e",
    "NCC" = "#2ca02c"
  )) +
  scale_color_manual(values = c(
    "LCC" = "#1f77b4",
    "RCC" = "#ff7f0e",
    "NCC" = "#2ca02c"
  )) +
  labs(
    x = "Aortic Cusp",
    y = expression("Calcification Volume ("*mm^3*")")
  ) +
  theme_classic(base_size = 10) +
  theme(
    legend.position = "none",
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )

print(leaflet_plot)



ggsave("leaflet_calcification_boxplot.pdf", leaflet_plot, width = 3.4, height = 3.0)
ggsave("leaflet_calcification_boxplot.png", leaflet_plot, width = 3.4, height = 3.0, dpi = 300)

# Test correlations with leaflet-specific calcification 
cor.test(data$LCC_calc, data$SAVI, method = "spearman") # Non-significant
cor.test(data$NCC_calc, data$SAVI, method = "spearman") # Significant relationship, p = 0.004
cor.test(data$RCC_calc, data$SAVI, method = "spearman") # Non-significant

# Compute correlation
r <- cor(data$NCC_calc, data$SAVI, use = "complete.obs", method = "spearman")

ggplot(data, aes(x = NCC_calc, y = SAVI)) +
  geom_point(size = 3, color = "steelblue") +
  geom_smooth(method = "lm", color = "firebrick",
              se = TRUE, linetype = "solid") +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0("r = ", round(r, 2)),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    title = "Relationship between NCC Calcification and SAVI",
    x = "NCC Calcification [mm^3]",
    y = "SAVI"
  ) +
  theme_minimal(base_size = 14)

### IEEE NCC CALC AGAINST SAVI

# Compute Spearman correlation
test <- cor.test(data$NCC_calc, data$SAVI, method = "pearson")
r <- unname(test$estimate)
p <- test$p.value

ggplot(data, aes(x = NCC_calc, y = SAVI)) +
  geom_point(
    size = 2,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",   # consistent blue regression line
    linewidth = 0.8,
    linetype = "solid"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "r = ", round(r, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("NCC Calcification ("*mm^3*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )


#### TO MAKE IT FOR IEEE 

leaflet_long <- data %>%
  select(LCC_calc, NCC_calc, RCC_calc, SAVI) %>%
  pivot_longer(
    cols = c(LCC_calc, NCC_calc, RCC_calc),
    names_to = "Leaflet",
    values_to = "CalcVolume"
  ) %>%
  mutate(
    Leaflet = factor(Leaflet,
                     levels = c("NCC_calc", "LCC_calc", "RCC_calc"),
                     labels = c("NCC", "LCC", "RCC"))
  )

leaflet_plot <- ggplot(leaflet_long, aes(x = Leaflet, y = CalcVolume, fill = Leaflet)) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    color = "black",
    linewidth = 1
  ) +
  geom_jitter(
    aes(color = Leaflet),
    width = 0.12,
    size = 2,
    alpha = 0.7,
    show.legend = FALSE
  ) +
  
  labs(
    x = "Leaflet",
    y = expression("Calcification Volume ("*mm^3*")")
  ) +
  
  scale_fill_manual(values = c(
    "NCC" = "#D55E00",
    "LCC" = "#0072B2",
    "RCC" = "#F0E442"
  )) +
  scale_color_manual(values = c(
    "NCC" = "#D55E00",
    "LCC" = "#0072B2",
    "RCC" = "#F0E442"
  )) +
  
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),   # ← same as your previous plots
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5),
    legend.position = "none"
  )

print(leaflet_plot)

# Export
ggsave("leaflet_calcification_boxplot.pdf",
       leaflet_plot, width = 8, height = 6)




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

data_long <- data %>%
  pivot_longer(
    cols = c(total_central_calc, total_peripheral_calc),
    names_to = "Location",
    values_to = "Calcification"
  ) %>%
  mutate(
    Location = factor(
      Location,
      levels = c("total_central_calc", "total_peripheral_calc"),
      labels = c("Central", "Peripheral")
    )
  )

ggplot(data_long, aes(x = Location, y = Calcification, fill = Location)) +
  geom_violin(
    width = 0.8,
    alpha = 0.6,
    color = "black",
    linewidth = 0.4,
    trim = FALSE
  ) +
  geom_jitter(
    aes(color = Location),
    width = 0.12,
    size = 1.3,
    alpha = 0.7,
    show.legend = FALSE
  ) +
  scale_fill_manual(values = c(
    "Central" = "#D55E00",
    "Peripheral" = "#0072B2"
  )) +
  scale_color_manual(values = c(
    "Central" = "#D55E00",
    "Peripheral" = "#0072B2"
  )) +
  labs(
    x = "Valve Region",
    y = "Calcification Volume"
  ) +
  theme_classic(base_size = 10) +
  theme(
    legend.position = "none",
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )

# IEEE FORMAT
# Check the actual values first
unique(data_long$Location)

# Example: if the values are "central" and "peripheral"
data_long$Location <- factor(
  data_long$Location,
  levels = c("central", "peripheral")
)

ggplot(data_long, aes(x = Location, y = Calcification, fill = Location)) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    color = "black",
    linewidth = 1
  ) +
  geom_jitter(
    width = 0.12,
    size = 2,
    alpha = 0.7
  ) +
  
  scale_x_discrete(labels = c(
    "Central Valve Region",
    "Peripheral Valve Region"
  )) +
  
  labs(
    x = "Location",
    y = expression("Calcification Volume ("*mm^3*")")
  ) +
  
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5),
    legend.position = "none"
  )

wilcox.test(data$total_peripheral_calc,
            data$total_central_calc,
            paired = FALSE,           # unpaired
            alternative = "two.sided", # test for difference in either direction
            na.action = na.omit)       # ignore missing values

data$peripheral_calc_density <- data$LCC_peripheral_area_mm2 + data$NCC_peripheral_area_mm2   + data$RCC_peripheral_area_mm2
data$central_calc_density <- data$LCC_central_area_mm2 + data$NCC_central_area_mm2   + data$RCC_central_area_mm2

# Gather the two variables into long format
data_long <- data %>%
  pivot_longer(
    cols = c(peripheral_calc_density, central_calc_density),
    names_to = "Location",
    values_to = "Calcification"
  )

# Plot side-by-side boxplots
ggplot(data_long, aes(x = Location, y = Calcification, fill = Location)) +
  geom_boxplot(width = 0.6, outlier.color = "red", outlier.shape = 16) +
  scale_fill_manual(values = c("steelblue", "firebrick")) +
  labs(
    title = "Total Central vs Peripheral Calcification Density",
    x = "Location",
    y = "Calcification Volume"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

wilcox.test(data$peripheral_calc_density,
            data$central_calc_density,
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

# Compute Spearman correlation
test <- cor.test(data$total_central_calc, data$SAVI, method = "spearman")

label_text <- paste0(
  "ρ = ", round(test$estimate, 2),
  "\np = ", signif(test$p.value, 2)
)

# Compute Spearman correlation
test <- cor.test(data$total_central_calc, data$SAVI, method = "spearman")

label_text <- paste0(
  "ρ = ", round(test$estimate, 2),
  "\np = ", signif(test$p.value, 2)
)

# IEEE PLOT CENTRAL CALCIFICATION


# Compute Spearman correlation
test <- cor.test(data$total_central_calc, data$SAVI, method = "spearman")
rho <- unname(test$estimate)
p <- test$p.value

ggplot(data, aes(x = total_central_calc, y = SAVI)) +
  geom_point(
    size = 2,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",   # same blue as your other plots
    linewidth = 0.8,
    linetype = "solid"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "\u03c1 = ", round(rho, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("Total Central Calcification ("*mm^3*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )

# IEEE FORMAT FOR PERIPHERAL NCC 

# Compute Spearman correlation
test <- cor.test(data$total_peripheral_calc, data$SAVI, method = "spearman")
rho <- unname(test$estimate)
p <- test$p.value

ggplot(data, aes(x = peripheral_NCC_calc, y = SAVI)) +
  geom_point(
    size = 2,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",   # changed from black → blue
    linewidth = 0.8,
    linetype = "solid"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "r = ", round(rho, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("Peripheral NCC Calcification ("*mm^3*")"),  # keep your variable name but same style
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )



scale_fill_manual(values = c("#1f77b4", "#ff7f0e"))

# Compute correlation and p-value
test <- cor.test(data$total_central_calc, data$SAVI, use = "complete.obs")
r <- test$estimate
p <- test$p.value

ggplot(data_long, aes(x = Location, y = Calcification, fill = Location)) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    color = "black",
    linewidth = 1
  ) +
  geom_jitter(
    width = 0.12,
    size = 2,
    alpha = 0.7
  ) +
  
  scale_fill_manual(values = c("#1f77b4", "#ff7f0e")) +
  
  scale_x_discrete(labels = c(
    "Central Valve Region",
    "Peripheral Valve Region"
  )) +
  
  labs(
    x = "Location",
    y = expression("Calcification Volume ("*mm^3*")")
  ) +
  
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5),
    legend.position = "none"
  )



# ------------------ DENSITY PERIPHERAL AND CENTRAL ------------

shapiro.test(data$total_central_calc) # Not normally distributed, p< 0.001

# Spearman correlation
test <- cor.test(data$total_central_calc, data$SAVI, method = "spearman")
rho <- unname(test$estimate)
p <- test$p.value

ggplot(data, aes(x = total_central_calc, y = SAVI)) +
  geom_point(size = 2, color = "black") +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",
    linewidth = 0.8
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "\u03c1 = ", round(rho, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("Central Region Calcification ("*mm^3*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )


shapiro.test(data$total_peripheral_calc) # normally distributed, p< 0.001

# Spearman correlation
test <- cor.test(data$total_peripheral_calc, data$SAVI, method = "pearson")
r <- unname(test$estimate)
p <- test$p.value

ggplot(data, aes(x = total_peripheral_calc, y = SAVI)) +
  geom_point(size = 2, color = "black") +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",
    linewidth = 0.8
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "r = ", round(r, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("Peripheral Region Calcification ("*mm^3*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )











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
shapiro.test(data$peripheral_NCC_calc) # normal

shapiro.test(data$central_RCC_calc) # not normal, = 0 .001
hist(data$central_RCC_calc)
shapiro.test(data$peripheral_RCC_calc) #not normal

shapiro.test(data$central_LCC_calc) # not normal
shapiro.test(data$peripheral_LCC_calc) # not normal, = 0.001
hist(data$peripheral_LCC_calc)




table_regions <- mean_sd_table_ieee %>%
  gt() %>%
  tab_header(
    title = "Table X",
    subtitle = "Relative Distribution of Calcification Across the Six Valve Regions"
  ) %>%
  cols_label(
    Cusp = "Cusp",
    Region = "Region",
    Percentage = "Calcification (%)"
  ) %>%
  cols_align(
    align = "center",
    columns = everything()
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_options(
    table.border.top.width = px(0),
    table.border.bottom.width = px(0),
    table.font.size = px(10),
    data_row.padding = px(2),
    heading.padding = px(2)
  ) %>%
  tab_footnote(
    footnote = "Values are reported as mean ± standard deviation.",
    locations = cells_title(groups = "subtitle")
  )

table_regions

writeLines(
  as_latex(table_regions),
  "H:/DATA/Afstuderen/3.Data/figures/Table_regions_distribution.tex"
)

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


# ------------------------------------------------SHOWCASING SIGNIFICANT WITHIN_LEAFLET EFFECTS ----------------

plot_spearman <- function(data, x_var, y_var, x_label, y_label, title) {
  
  test <- cor.test(data[[x_var]], data[[y_var]], method = "spearman")
  
  label_text <- paste0(
    "ρ = ", round(test$estimate, 2),
    "\np = ", signif(test$p.value, 2)
  )
  
  ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(size = 3, color = "steelblue") +
    geom_smooth(method = "lm", color = "firebrick", se = TRUE) +
    annotate("text",
             x = Inf, y = Inf,
             label = label_text,
             hjust = 1.1, vjust = 1.5,
             size = 5) +
    labs(title = title, x = x_label, y = y_label) +
    theme_minimal(base_size = 14)
}

plot_spearman(data, "central_NCC_calc", "SAVI",
              "Central NCC Calcification [mm³]", "SAVI",
              "SAVI vs Central NCC Calcification")

plot_spearman(data, "peripheral_NCC_calc", "SAVI",
              "Peripheral NCC Calcification [mm³]", "SAVI",
              "SAVI vs Peripheral NCC Calcification")

plot_spearman(data, "central_RCC_calc", "SAVI",
              "Central RCC Calcification [mm³]", "SAVI",
              "SAVI vs Central RCC Calcification")

test <- cor.test(data$central_NCC_calc, data$SAVI, method = "spearman")
rho <- unname(test$estimate)
p <- test$p.value

ggplot(data, aes(x = central_NCC_calc, y = SAVI)) +
  geom_point(
    size = 2,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",   # ← blue line
    linewidth = 0.8,
    linetype = "solid"
  )  +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "\u03c1 = ", round(rho, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("Central NCC Calcification ("*mm^3*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )

# Compute correlation
test <- cor.test(data$peripheral_NCC_calc, data$SAVI, use = "complete.obs")
r <- test$estimate
p <- test$p.value

ggplot(data, aes(x = peripheral_NCC_calc, y = SAVI)) +
  geom_point(
    size = 2,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "black",
    linewidth = 0.8,
    linetype = "solid"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "r = ", round(r, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("Peripheral NCC Calcification ("*mm^3*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )

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


# -------------------------------------------------- SPLITTING THE SAVI -------------------------

data$SAVI_group <- ifelse(
  data$SAVI < 0.7,
  "Low SAVI",
  "High SAVI"
)

table(data$SAVI_group)

library(ggpubr)

data$central_ratio <- data$total_central_calc / data$total_peripheral_calc

ggplot(data, aes(x = SAVI_group, y = total_central_calc, fill = SAVI_group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, size = 2, alpha = 0.7) +
  stat_compare_means(method = "t.test", label = "p.format") +
  labs(
    title = "Central Calcification Proportion by SAVI Group",
    x = "SAVI Group",
    y = "Total Central Calc "
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

data$peripheral_prop <- data$total_peripheral_calc / data$calc_volume

ggplot(data, aes(x = SAVI_group, y = total_peripheral_calc, fill = SAVI_group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, size = 2, alpha = 0.7) +
  stat_compare_means(method = "t.test", label = "p.format") +
  labs(
    title = "Peripheral Calcification by SAVI Group",
    x = "SAVI Group",
    y = "Total Peripheral Calcification [mm³]"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# Both non significant

# -------------------------- CALCIFICATION DENSITY ------------------------------

# ---------------- CALCIFICATION DENSITY ----------------

data <- data %>%
  mutate(
    # NCC
    NCC_total_density = NCC_calc / NCC_total_area_mm2,
    NCC_central_density = central_NCC_calc / NCC_central_area_mm2,
    NCC_peripheral_density = peripheral_NCC_calc / NCC_peripheral_area_mm2,
    
    # LCC
    LCC_total_density = LCC_calc / LCC_total_area_mm2,
    LCC_central_density = central_LCC_calc / LCC_central_area_mm2,
    LCC_peripheral_density = peripheral_LCC_calc / LCC_peripheral_area_mm2,
    
    # RCC
    RCC_total_density = RCC_calc / RCC_total_area_mm2,
    RCC_central_density = central_RCC_calc / RCC_central_area_mm2,
    RCC_peripheral_density = peripheral_RCC_calc / RCC_peripheral_area_mm2
  )

# ---------------- CALCIFICATION DENSITY ----------------

data <- data %>%
  mutate(
    # NCC
    NCC_total_density = NCC_calc / NCC_total_area_mm2,
    NCC_central_density = central_NCC_calc / NCC_central_area_mm2,
    NCC_peripheral_density = peripheral_NCC_calc / NCC_peripheral_area_mm2,
    
    # LCC
    LCC_total_density = LCC_calc / LCC_total_area_mm2,
    LCC_central_density = central_LCC_calc / LCC_central_area_mm2,
    LCC_peripheral_density = peripheral_LCC_calc / LCC_peripheral_area_mm2,
    
    # RCC
    RCC_total_density = RCC_calc / RCC_total_area_mm2,
    RCC_central_density = central_RCC_calc / RCC_central_area_mm2,
    RCC_peripheral_density = peripheral_RCC_calc / RCC_peripheral_area_mm2
  )

density_long <- data %>%
  select(NCC_total_density, LCC_total_density, RCC_total_density) %>%
  pivot_longer(
    cols = everything(),
    names_to = "Cusp",
    values_to = "Density"
  ) %>%
  mutate(
    Cusp = recode(Cusp,
                  NCC_total_density = "NCC",
                  LCC_total_density = "LCC",
                  RCC_total_density = "RCC")
  )

ggplot(density_long, aes(x = Cusp, y = Density, fill = Cusp)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.12, size = 2, alpha = 0.7) +
  labs(
    title = "Leaflet-specific Calcification Density",
    x = "Cusp",
    y = "Calcification Density [mm³/mm²]"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

friedman.test(
  y = as.matrix(data[, c("NCC_total_density", "LCC_total_density", "RCC_total_density")])
)

pairwise_results <- data.frame(
  Comparison = c("NCC vs LCC", "NCC vs RCC", "LCC vs RCC"),
  p_value = c(
    wilcox.test(data$NCC_total_density, data$LCC_total_density, paired = TRUE)$p.value,
    wilcox.test(data$NCC_total_density, data$RCC_total_density, paired = TRUE)$p.value,
    wilcox.test(data$LCC_total_density, data$RCC_total_density, paired = TRUE)$p.value
  )
)

# Bonferroni correction
pairwise_results$p_adjusted <- p.adjust(pairwise_results$p_value, method = "bonferroni")

pairwise_results


# ------------------------------- CALCIUM DENSITY PER LEAFLET ---------------------------------

# SAVI
cor.test(data$NCC_total_density, data$SAVI, method = "pearson") # Significant
cor.test(data$LCC_total_density, data$SAVI, method = "pearson") # non significant
cor.test(data$RCC_total_density, data$SAVI, method = "pearson") # 

# Compute Spearman correlation
test <- cor.test(data$NCC_total_density, data$SAVI, method = "spearman")

label_text <- paste0(
  "ρ = ", round(test$estimate, 2),
  "\np = ", signif(test$p.value, 2)
)

ggplot(data, aes(x = NCC_total_density, y = SAVI)) +
  geom_point(size = 3, color = "steelblue") +
  geom_smooth(method = "lm", color = "firebrick", se = TRUE) +  # visual trend
  annotate(
    "text",
    x = Inf, y = Inf,
    label = label_text,
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    title = "Relationship between NCC Calcification Density and SAVI",
    x = "NCC Calcification Density [mm³/mm²]",
    y = "SAVI"
  ) +
  theme_minimal(base_size = 14)

# IEEE PLOT NCC DENSITY

# Compute correlation and p-value
shapiro.test(data$NCC_total_density) 
test <- cor.test(data$NCC_total_density, data$SAVI, use = "complete.obs")
r <- test$estimate
p <- test$p.value

ggplot(data, aes(x = NCC_total_density, y = SAVI)) +
  geom_point(
    size = 2,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#0072B2",   # ← blue line
    linewidth = 0.8,
    linetype = "solid"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste0(
      "r = ", round(r, 2),
      ", p ", ifelse(p < 0.001, "< 0.001", paste0("= ", round(p, 3)))
    ),
    hjust = 1.1, vjust = 1.5,
    size = 5
  ) +
  labs(
    x = expression("NCC Calcification ("*mm^3/mm^2*")"),
    y = "SAVI"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1),
    axis.ticks = element_line(linewidth = 0.5)
  )


# Model for SAVI
SAVI_lm_density <- lm(
  SAVI ~ NCC_total_density + LCC_total_density + RCC_total_density,
  data = data
)

summary(SAVI_lm_density)


# PEAK_VELOCITY
cor.test(data$NCC_total_density, data$peak_velocity, method = "pearson") # # non significant
cor.test(data$LCC_total_density, data$peak_velocity, method = "pearson") # non significant
cor.test(data$RCC_total_density, data$peak_velocity, method = "pearson") ## non significant# non significant

# MEANRR
cor.test(data$NCC_total_density, data$meanRR, method = "pearson") # # non significant
cor.test(data$LCC_total_density, data$meanRR, method = "pearson") # non significant
cor.test(data$RCC_total_density, data$meanRR, method = "pearson") # # non significant

# AVApre
cor.test(data$NCC_total_density, data$AVApre, method = "pearson") # # non significant
cor.test(data$LCC_total_density, data$AVApre, method = "pearson") # # non significant
cor.test(data$RCC_total_density, data$AVApre, method = "pearson") # # non significant


# ------------------- WITHIN LEAFLET ANALYSIS -------------------------

cor.test(data$NCC_total_density, data$AVApre, method = "pearson") # # non significant
cor.test(data$LCC_total_density, data$AVApre, method = "pearson") # # non significant
cor.test(data$RCC_total_density, data$AVApre, method = "pearson") # # non significant

region_density_vars <- c(
  "NCC_central_density", "NCC_peripheral_density",
  "LCC_central_density", "LCC_peripheral_density",
  "RCC_central_density", "RCC_peripheral_density"
)

outcomes <- c("peak_velocity", "meanRR", "AVApre", "SAVI")

results_list <- list()

for (outcome in outcomes) {
  res <- lapply(region_density_vars, function(var) {
    
    test <- cor.test(data[[var]], data[[outcome]], method = "spearman")
    
    data.frame(
      Outcome = outcome,
      Region = var,
      rho = test$estimate,
      p_value = test$p.value
    )
  })
  
  results_list[[outcome]] <- do.call(rbind, res)
}

results_region_density <- do.call(rbind, results_list)

# Multiple testing correction (important!)
results_region_density$p_adjusted <- p.adjust(results_region_density$p_value, method = "BH")

results_region_density


# %% LINEAR REGRESSION MODEL

# Model for SAVI
SAVI_lm_density <- lm(
  SAVI ~ NCC_total_density + LCC_total_density + RCC_total_density,
  data = data
)

summary(SAVI_lm_density)

par(mfrow = c(2,2))
plot(SAVI_lm_density)
library(car)
vif(SAVI_lm_density)

shapiro.test(data$RCC_total_density) # not normal
shapiro.test(data$NCC_total_density) # normal
shapiro.test(data$LCC_total_density) # not normal


### HEATMAP

library(ggplot2)

heatmap_data <- data.frame(
  Cusp = c("NCC","LCC","RCC","NCC","RCC","LCC"),
  Region = c("Peripheral","Peripheral","Peripheral","Central","Central","Central"),
  Mean = c(29.305940,22.181112,20.812538,10.935880,8.565545,8.198985)
)

ggplot(heatmap_data, aes(x = Region, y = Cusp, fill = Mean)) +
  geom_tile(color = "black") +
  
  # Add values inside tiles
  geom_text(aes(label = round(Mean, 1)), size = 4) +
  
  scale_fill_gradient(
    low = "white",
    high = "black"
  ) +
  
  labs(
    x = "Region",
    y = "Cusp"
  ) +
  
  theme_classic(base_size = 10) +
  theme(
    legend.title = element_blank(),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

# -------------- SPLIT CENTRAL CALCIFICATION FOR THE SAVI GROUPS

data <- data %>%
  mutate(
    SAVI_group = ifelse(SAVI < 0.7, "Severe", "Non-severe")
  )

shapiro.test(data$total_central_calc[data$SAVI_group == "Severe"]) # Normally distributed
shapiro.test(data$total_central_calc[data$SAVI_group == "Non-severe"])  # Normally distributed


ggplot(data, aes(x = SAVI_group, y = total_central_calc, fill = SAVI_group)) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    color = "black",
    linewidth = 1
  ) +
  geom_jitter(
    width = 0.12,
    size = 2,
    alpha = 0.7
  ) +
  scale_fill_manual(values = c(
    "Severe" = "#ff7f0e",
    "Non-severe" = "#1f77b4"
  )) +
  labs(
    x = "Stenosis Severity (SAVI)",
    y = expression("Central Calcification ("*mm^3*")")
  ) +
  theme_classic(base_size = 10) +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    axis.line = element_line(linewidth = 1)
  )

wilcox.test(total_central_calc ~ SAVI_group, data = data)



# ---------------------- MAKE A HEATMAP ----------------------------

c

median(data$peripheral_NCC_calc, na.rm = TRUE)


# BETWEEN-LEAFLET

leaflet_vars <- c("NCC_total_density", "LCC_total_density", "RCC_total_density")

leaflet_median_iqr_table <- data.frame(
  Leaflet = leaflet_vars,
  Median = sapply(data[, leaflet_vars], median, na.rm = TRUE),
  Q1 = sapply(data[, leaflet_vars], quantile, probs = 0.25, na.rm = TRUE),
  Q3 = sapply(data[, leaflet_vars], quantile, probs = 0.75, na.rm = TRUE)
)

leaflet_median_iqr_table <- leaflet_median_iqr_table %>%
  mutate(
    Leaflet = factor(Leaflet, levels = c("NCC_total_density", "LCC_total_density", "RCC_total_density")),
    Median_IQR = sprintf("%.1f [%.1f–%.1f]", Median, Q1, Q3)
  ) %>%
  select(Leaflet, Median, Q1, Q3, Median_IQR)

leaflet_median_iqr_table



# CENTRAL VS. PERIPHERAL

vars <- c("total_central_calc", "total_peripheral_calc")

central_peripheral_table <- data.frame(
  Variable = vars,
  Median = sapply(data[, vars], median, na.rm = TRUE),
  Q1 = sapply(data[, vars], quantile, probs = 0.25, na.rm = TRUE),
  Q3 = sapply(data[, vars], quantile, probs = 0.75, na.rm = TRUE)
)

# Clean naming
central_peripheral_table$Region <- ifelse(
  grepl("central", central_peripheral_table$Variable),
  "Central", "Peripheral"
)

# Final formatting
central_peripheral_table <- central_peripheral_table %>%
  mutate(
    Region = factor(Region, levels = c("Central", "Peripheral")),
    Median_IQR = sprintf("%.1f [%.1f–%.1f]", Median, Q1, Q3)
  ) %>%
  arrange(Region) %>%
  select(Region, Median, Q1, Q3, Median_IQR)

central_peripheral_table


## MAKE THE HEATMAP ITSELF!!

# Example input table:
# heatmap_data should contain:
#   Cusp:   NCC, RCC, LCC
#   Region: Central, Peripheral
#   Median: numeric value

library(ggplot2)
library(dplyr)

heatmap_data <- data.frame(
  Cusp = c("NCC", "RCC", "LCC", "NCC", "RCC", "LCC"),
  Region = c("Central", "Central", "Central", "Peripheral", "Peripheral", "Peripheral"),
  Median = c(35, 20, 28, 55, 42, 48)
)

heatmap_data$Cusp <- factor(heatmap_data$Cusp, levels = c("NCC", "RCC", "LCC"))
heatmap_data$Region <- factor(heatmap_data$Region, levels = c("Central", "Peripheral"))

# Simple sector function
make_sector <- function(start_deg, end_deg, r_inner, r_outer, n = 200) {
  theta_outer <- seq(start_deg, end_deg, length.out = n) * pi / 180
  theta_inner <- seq(end_deg, start_deg, length.out = n) * pi / 180
  
  x <- c(r_outer * cos(theta_outer), r_inner * cos(theta_inner))
  y <- c(r_outer * sin(theta_outer), r_inner * sin(theta_inner))
  
  data.frame(x = x, y = y)
}

# Each cusp takes exactly one third of the circle
sector_defs <- data.frame(
  Cusp = c("NCC", "RCC", "LCC"),
  start = c(210, 330, 90),
  end   = c(330, 450, 210)
)

# Build polygons
plot_data <- list()
id <- 1

for (i in seq_len(nrow(heatmap_data))) {
  cusp_i <- as.character(heatmap_data$Cusp[i])
  region_i <- as.character(heatmap_data$Region[i])
  median_i <- heatmap_data$Median[i]
  
  sector_i <- sector_defs %>% filter(Cusp == cusp_i)
  
  if (region_i == "Central") {
    r_inner <- 0.00
    r_outer <- 0.50
  } else {
    r_inner <- 0.50
    r_outer <- 1.00
  }
  
  poly <- make_sector(
    start_deg = sector_i$start,
    end_deg   = sector_i$end,
    r_inner   = r_inner,
    r_outer   = r_outer
  )
  
  poly$Cusp <- cusp_i
  poly$Region <- region_i
  poly$Median <- median_i
  poly$id <- id
  
  plot_data[[id]] <- poly
  id <- id + 1
}

plot_data <- bind_rows(plot_data)

# Label positions adjusted to your actual radii
label_data <- data.frame(
  Cusp = c("NCC", "RCC", "LCC", "NCC", "RCC", "LCC"),
  Region = c("Central", "Central", "Central", "Peripheral", "Peripheral", "Peripheral"),
  angle = c(270, 30, 150, 270, 30, 150),
  r = c(0.25, 0.25, 0.25, 0.75, 0.75, 0.75)
) %>%
  left_join(heatmap_data, by = c("Cusp", "Region")) %>%
  mutate(
    x = r * cos(angle * pi / 180),
    y = r * sin(angle * pi / 180),
    label = sprintf("%.1f", Median)
  )

ggplot() +
  geom_polygon(
    data = plot_data,
    aes(x = x, y = y, group = id, fill = Median),
    color = "black",
    linewidth = 0.6
  ) +
  geom_text(
    data = label_data,
    aes(x = x, y = y, label = label),
    size = 4
  ) +
  annotate("text", x = 0, y = -1.15, label = "NCC", size = 5, fontface = "bold") +
  annotate("text", x = 1.08, y = 0.05, label = "RCC", size = 5, fontface = "bold") +
  annotate("text", x = -1.08, y = 0.05, label = "LCC", size = 5, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "red", name = "Median") +
  coord_fixed() +
  theme_void() +
  theme(
    legend.position = "right"
  )

