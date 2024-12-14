# Libraries
library(tidyverse)
library(data.table)
library(caret)
library(syuzhet)
library(glmnet)

# Data Import and Sampling
df <- read_csv("datasets/merged_data.csv")
df <- df[sample(nrow(df), 1000), ]

# Genre Splitting and Imputation
df <- df %>%
  mutate(
    Tgenres = str_split(TomatoesGenres, "&"),
    Mgenres = str_split(MetacriticGenres, "\\|")
  ) %>%
  unnest(Tgenres) %>%
  unnest(Mgenres) %>%
  mutate(
    Tgenres = ifelse(is.na(Tgenres), "Unknown", Tgenres),
    Mgenres = ifelse(is.na(Mgenres), "Unknown", Mgenres)
  )

# Sentiment Score Calculation
df$review_detail <- as.character(df$review_detail)
sentiments <- get_nrc_sentiment(df$review_detail)
df$sentiment_score <- sentiments$positive - sentiments$negative

# Keyword Extraction
top_keywords <- c("nonsense", "rewatch", "sofa", "remove", "xmas")
for (keyword in top_keywords) {
  df[[keyword]] <- ifelse(grepl(keyword, df$review_detail, ignore.case = TRUE), 1, 0)
}
summary(df)
# Ensure `OtherKeywords` exists
keyword_columns <- colnames(df)[colnames(df) %in% top_keywords]
if (!"OtherKeywords" %in% colnames(df)) {
  df$OtherKeywords <- rowSums(df[keyword_columns], na.rm = TRUE)
}

# Define predictors and target
predictors <- c("Mgenres", "Tgenres", "NashSource", "NashProductionMethod", 
                "NashProductionBudget", "NashYear","helpful","review_date",
                "NashRunningTimeMinutes", "NashMpaaRating", top_keywords, 
                "OtherKeywords", "sentiment_score")
target <- "NashDomesticBoxOffice"

# Check Missing Predictors
missing_columns <- setdiff(predictors, colnames(df))
if (length(missing_columns) > 0) {
  cat("The following columns are missing and will be excluded from predictors:\n")
  print(missing_columns)
}
predictors <- predictors[predictors %in% colnames(df)]

# Select Predictors and Target
X <- df %>% select(all_of(predictors))
y <- df[[target]]

# Train-Test Split
set.seed(42)
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

# Preprocessing: Handle Missing Columns
missing_in_train <- setdiff(colnames(X_test), colnames(X_train))
missing_in_test <- setdiff(colnames(X_train), colnames(X_test))
if (length(missing_in_train) > 0) {
  for (col in missing_in_train) {
    X_train[[col]] <- 0
  }
}
if (length(missing_in_test) > 0) {
  for (col in missing_in_test) {
    X_test[[col]] <- 0
  }
}


# Ensure Consistent Column Order
X_test <- X_test[, colnames(X_train), drop = FALSE]

# Remove Near-Zero Variance Columns
nzv <- nearZeroVar(X_train, saveMetrics = TRUE)
X_train <- X_train[, !nzv$zeroVar]
X_test <- X_test[, !nzv$zeroVar]

# Apply One-Hot Encoding
dummy_encoder <- dummyVars(~ ., data = X_train)
X_train_encoded <- predict(dummy_encoder, newdata = X_train)
X_test_encoded <- predict(dummy_encoder, newdata = X_test)

# Ensure Column Alignment
common_columns <- intersect(colnames(X_train_encoded), colnames(X_test_encoded))
X_train_encoded <- X_train_encoded[, common_columns]
X_test_encoded <- X_test_encoded[, common_columns]

# Preprocessing: Centering and Scaling
preprocess_pipeline <- preProcess(as.data.frame(X_train_encoded), method = c("center", "scale"))
X_train_processed <- predict(preprocess_pipeline, as.data.frame(X_train_encoded))
X_test_processed <- predict(preprocess_pipeline, as.data.frame(X_test_encoded))

# Model Training
lasso_model <- cv.glmnet(as.matrix(X_train_processed), y_train, alpha = 1, lambda = 10^seq(-3, 3, length = 100))

# Predictions
y_pred_train <- predict(lasso_model, as.matrix(X_train_processed), s = "lambda.min")
y_pred_test <- predict(lasso_model, as.matrix(X_test_processed), s = "lambda.min")

# Calculate Metrics
train_mse <- mean((y_train - y_pred_train)^2)
train_mae <- mean(abs(y_train - y_pred_train))
train_r2 <- 1 - sum((y_train - y_pred_train)^2) / sum((y_train - mean(y_train))^2)

test_mse <- mean((y_test - y_pred_test)^2)
test_mae <- mean(abs(y_test - y_pred_test))
test_r2 <- 1 - sum((y_test - y_pred_test)^2) / sum((y_test - mean(y_test))^2)

# Print Results
cat("Train MSE : ", train_mse, "\n")
cat("Train MAE : ", train_mae, "\n")
cat("Train R2  : ", train_r2, "\n\n")

cat("Test MSE  : ", test_mse, "\n")
cat("Test MAE  : ", test_mae, "\n")
cat("Test R2   : ", test_r2, "\n")

