# ==========================================
# HOMICIDE MULTIVARIATE PROJECT - FINAL R CODE (With Comments)
# ==========================================

# Load required packages
library(dplyr)
library(readr)
library(fastDummies)
library(ggplot2)
library(factoextra)
library(caret)
library(psych)
library(rpart)
library(rpart.plot)
library(gridExtra)
library(vcd)
library(forcats)

# ------------------------
# 1. Load and Clean Data
# ------------------------
# Read dataset
df <- read_csv("database.csv")

# Rename columns (replace spaces with underscores)
colnames(df) <- gsub(" ", "_", colnames(df))

df <- df %>% rename(
  record_id = Record_ID,
  crime_solved = Crime_Solved,
  weapon = Weapon,
  relationship = Relationship,
  victim_sex = Victim_Sex,
  victim_age = Victim_Age,
  victim_race = Victim_Race,
  victim_ethnicity = Victim_Ethnicity,
  perpetrator_sex = Perpetrator_Sex,
  perpetrator_age = Perpetrator_Age,
  perpetrator_race = Perpetrator_Race,
  perpetrator_ethnicity = Perpetrator_Ethnicity,
  victim_count = Victim_Count,
  perpetrator_count = Perpetrator_Count,
  year = Year,
  state = State
)

# Convert age fields to numeric
df$victim_age <- as.numeric(df$victim_age)
df$perpetrator_age <- as.numeric(df$perpetrator_age)

# Filter invalid or unknown values
df_clean <- df %>%
  filter(
    !is.na(victim_age) & victim_age > 0 & victim_age <= 110,
    !is.na(perpetrator_age) & perpetrator_age > 0 & perpetrator_age <= 110,
    victim_count >= 1,
    perpetrator_count >= 1,
    weapon != "Unknown",
    relationship != "Unknown",
    victim_sex %in% c("Male", "Female"),
    perpetrator_sex %in% c("Male", "Female"),
    crime_solved %in% c("Yes", "No")
  )

# ------------------------
# 2. PCA (Principal Component Analysis)
# ------------------------
# Select numeric variables for PCA
df_pca_input <- df_clean %>%
  select(victim_age, perpetrator_age, victim_count, perpetrator_count)

# Scale the data
df_pca_scaled <- scale(df_pca_input)

# Run PCA and plot scree/biplot
pca_result <- prcomp(df_pca_scaled, scale. = TRUE)
fviz_eig(pca_result, addlabels = TRUE, barfill = "skyblue")
fviz_pca_biplot(pca_result, repel = TRUE,
                col.var = "red", col.ind = "gray40",
                title = "PCA Biplot: Homicide Features")

# ------------------------
# 3. Clustering (K-means)
# ------------------------
# Prepare data for clustering
df_cluster_base <- df_clean

df_cluster <- df_cluster_base %>%
  select(victim_age, perpetrator_age, victim_count, perpetrator_count, weapon, relationship, victim_sex, perpetrator_sex) %>%
  dummy_cols(select_columns = c("weapon", "relationship", "victim_sex", "perpetrator_sex"), remove_selected_columns = TRUE)

# Scale variables
df_cluster_scaled <- scale(df_cluster)

# Determine optimal number of clusters
fviz_nbclust(df_cluster_scaled, kmeans, method = "wss") + labs(title = "optimal number of clusters(Elbow Method)")
fviz_nbclust(df_cluster_scaled, kmeans, method = "silhouette") + labs(title = "Silhouette Method")

# Apply k-means clustering
set.seed(123)
kmeans_result <- kmeans(df_cluster_scaled, centers = 2, nstart = 25)
df_cluster_base$cluster <- as.factor(kmeans_result$cluster)

# Cluster visualization
fviz_cluster(kmeans_result, data = df_cluster_scaled, ellipse.type = "norm", geom = "point", main = "K-means Clustering of Homicide Cases")

# Cluster scatterplot by age
ggplot(df_cluster_base, aes(x = victim_age, y = perpetrator_age, color = cluster)) +
  geom_point(alpha = 0.5) + theme_minimal() + labs(title = "Clusters by Age")

# Cluster profiling
df_cluster_base %>% group_by(cluster) %>% summarise(
  cases = n(),
  avg_victim_age = mean(victim_age),
  avg_perp_age = mean(perpetrator_age),
  solved_rate = mean(crime_solved == "Yes"),
  top_weapon = names(sort(table(weapon), decreasing = TRUE))[1]
)

# ------------------------
# 4. Factor Analysis
# ------------------------
# Prepare and scale data
df_fa <- df_clean %>% select(victim_age, perpetrator_age, victim_count, perpetrator_count)
df_fa_scaled <- scale(df_fa)

# Determine number of factors
fa.parallel(df_fa_scaled, fa = "fa", n.iter = 100, show.legend = TRUE)

# Apply Factor Analysis
fa_result <- fa(df_fa_scaled, nfactors = 2, rotate = "varimax", fm = "ml")
print(fa_result$loadings)
fa.diagram(fa_result)

# ------------------------
# 5. Basic Descriptive Analysis
# ------------------------
# Weapon counts overall and by gender
by.weapon <- df_clean %>% group_by(weapon) %>% summarise(freq.by.weapon = n()) %>% arrange(desc(freq.by.weapon))
by.weapon.sex <- df_clean %>% group_by(weapon, victim_sex) %>% summarise(freq.by.weapon = n()) %>% rename(victim.sex = victim_sex)

# Weapon usage plot
Plot.weapon.used <- ggplot(by.weapon, aes(x = weapon, y = freq.by.weapon)) +
  geom_bar(stat = "identity", fill = "red", width = 0.5) + theme_light() +
  ggtitle("Weapon Used in Homicides") + labs(x = "Weapon", y = "Number of Incidents") +
  theme(axis.text.x = element_text(size = 8, angle = 90, hjust = 1))

# Weapon vs Gender plot
plot.Weapon.vs.gender <- ggplot(by.weapon.sex, aes(x = weapon, y = freq.by.weapon)) +
  geom_bar(stat = "identity", fill = "red", width = 0.5) + facet_wrap(~ victim.sex) +
  theme_light() + ggtitle("Weapon Used by Victim Gender") +
  labs(x = "Weapon", y = "Number of Incidents") +
  theme(axis.text.x = element_text(size = 6, angle = 90, hjust = 1))

grid.arrange(Plot.weapon.used, plot.Weapon.vs.gender, ncol = 1)

# Incidents by Year and State
by.year <- df_clean %>% group_by(year) %>% summarise(freq.year = n())
by.state <- df_clean %>% group_by(state) %>% summarise(freq.by.state = n()) %>% arrange(desc(freq.by.state))
by.state$state <- fct_inorder(by.state$state)

# Yearly trend plots
plot.homic.years <- ggplot(by.year, aes(x = as.numeric(year), y = freq.year)) +
  geom_point(size = 1.5, color = "blue") +
  geom_line(size = 0.5, color = "yellow3") + theme_light() +
  ggtitle("Number of Homicide Incidents per Year") +
  labs(x = "Year", y = "Number of Incidents") +
  theme(axis.text.x = element_text(size = 5, angle = 90, hjust = 0.5))

plot.by.state <- ggplot(by.state, aes(x = as.factor(state), y = freq.by.state)) +
  geom_bar(stat = "identity", fill = "darkred", width = 0.5) +
  geom_text(aes(label = freq.by.state), vjust = -0.3, size = 2.5, color = "black") +  # Add count labels
  theme_light() +
  ggtitle("Number of Homicide Incidents by State") +
  labs(x = "State", y = "Number of Incidents") +
  theme(axis.text.x = element_text(size = 6, angle = 90, hjust = 0.5))

grid.arrange(plot.homic.years, plot.by.state, ncol = 1)

# Boxplot: Victim age by weapon
ggplot(df_clean, aes(x = weapon, y = victim_age)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red") +
  theme_minimal() + labs(title = "Victim Age Distribution by Weapon Type", x = "Weapon Used", y = "Victim Age") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8))

# ------------------------
# 7. Mosaic Plot (Race + Ethnicity)
# ------------------------
# Prepare data for mosaic plot
race_eth_data <- df_clean %>%
  filter(
    !victim_race %in% c("Unknown", NA),
    !perpetrator_race %in% c("Unknown", NA),
    !victim_ethnicity %in% c("Unknown", NA),
    !perpetrator_ethnicity %in% c("Unknown", NA)
  ) %>%
  mutate(
    Victim_Group = paste(victim_race, victim_ethnicity, sep = "_"),
    Perp_Group = paste(perpetrator_race, perpetrator_ethnicity, sep = "_")
  )

# Contingency table
combined_table <- xtabs(~ Victim_Group + Perp_Group, data = race_eth_data)

# Expand window and plot mosaic
windows(width = 14, height = 10)
mosaic(
  combined_table,
  shade = TRUE,
  legend = TRUE,
  labeling_args = list(rot_labels = c(90, 0)),
  main = "Victim vs Perpetrator Race + Ethnicity"
)
