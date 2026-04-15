# ---------------- LOAD PACKAGES -------------

library(readxl)
library(gt)
library(dplyr)
library(ggplot2)
library(tidyr)
library(survival)
library(survminer)


# ---------------- LOAD & PREPROCESS DATA -----------------

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


# --------------- EXPLORATIVE DATA ANALYSIS ----------------------

#### OVERALL CALCIFICATION VOLUME


### AVR NEEDED

median_val <- median(data$calc_volume, na.rm = TRUE)
data$calc_group <- ifelse(data$calc_volume > median_val, "High", "Low")

surv_obj_avr <- Surv(time = data$Endpoint_days_AVR,
                     event = data$AVR_yes)

fit_avr <- survfit(surv_obj_avr ~ calc_group, data = data)

ggsurvplot(fit_avr,
           data = data,
           pval = TRUE,
           risk.table = TRUE,
           title = "Kaplan-Meier Curve for AVR")


## CARDIAC DEATH SURVIVAL

surv_obj_death <- Surv(time = data$Endpoint_days_death,
                       event = data$Cardiac_death)

fit_death <- survfit(surv_obj_death ~ calc_group, data = data)

ggsurvplot(fit_death,
           data = data,
           pval = TRUE,
           risk.table = TRUE,
           title = "Kaplan-Meier Curve for Cardiac Death")

cox_ncc <- coxph(Surv(Endpoint_days_AVR, AVR_yes) ~ log(calc_volume + 1),
                 data = data)

summary(cox_ncc) # Not significant



### NCC SPECIFIC

median_val_ncc <- median(data$NCC_calc, na.rm = TRUE)
data$calc_group_ncc <- ifelse(data$NCC_calc > median_val_ncc, "High", "Low")

surv_obj_avr <- Surv(time = data$Endpoint_days_AVR,
                     event = data$AVR_yes)

fit_avr_ncc <- survfit(surv_obj_avr ~ calc_group_ncc, data = data)

ggsurvplot(fit_avr_ncc,
           data = data,
           pval = TRUE,
           risk.table = TRUE,
           title = "Kaplan-Meier Curve for AVR (NCC calcification)")


cox_ncc <- coxph(Surv(Endpoint_days_AVR, AVR_yes) ~ log(NCC_calc + 1),
                 data = data)

summary(cox_ncc)

