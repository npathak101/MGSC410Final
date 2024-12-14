# Libraries
library(tidyverse)
library(data.table)
library(caret)
library(syuzhet)
library(glmnet)

setwd("/Users/nirvanipathak/Desktop/New Final")

# Data Import and Sampling
df <- read_csv("/Users/nirvanipathak/Desktop/New Final/merged_data.csv")
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




# Define UI
# Libraries
library(shiny)
library(shinythemes)
library(tidytext)

# Define UI
ui <- navbarPage(
  title = "Movie Reviews App",
  theme = shinytheme("flatly"),  # Add a modern theme
  tabPanel(
    "Home",
    fluidRow(
      column(
        4,
        wellPanel(
          h4("Variable Descriptions"),
          verbatimTextOutput("variableDescriptions")
        )
      ),
      column(
        8,
        h4("Descriptions"),
        tableOutput("varTable")
      )
    )
  ),
  tabPanel(
    "Predict Price",
    sidebarLayout(
      sidebarPanel(
        h4("Input Predictors"),
        numericInput("NashProductionBudget", "Production Budget ($M)", value = 50),
        numericInput("NashYear", "Release Year", value = 2020),
        numericInput("helpful", "Helpful Votes", value = 0),
        numericInput("NashRunningTimeMinutes", "Running Time (Minutes)", value = 120),
        numericInput("sentiment_score", "Sentiment Score", value = 0),
        actionButton("predict", "Predict", class = "btn btn-primary")
      ),
      mainPanel(
        h4("Prediction Result"),
        verbatimTextOutput("pricePrediction")
      )
    )
  ),
  tabPanel(
    "Analyze Reviews",
    sidebarLayout(
      sidebarPanel(
        h4("Filter Reviews"),
        selectInput("movie", "Select a Movie", choices = unique(df$movie)),
        checkboxInput("include_spoilers", "Include Spoilers?", value = FALSE),
        actionButton("analyze", "Analyze Reviews", class = "btn btn-primary")
      ),
      mainPanel(
        h4("Filtered Reviews"),
        tableOutput("reviewTable"),
        h4("Rating Distribution"),
        plotOutput("ratingDistribution", height = "300px")
      )
    )
  ),
  tabPanel(
    "Movie Info",
    sidebarLayout(
      sidebarPanel(
        h4("Search Movie Info"),
        textInput("movie_name", "Enter Movie Name", value = ""),
        actionButton("get_movie_info", "Get Movie Info", class = "btn btn-primary")
      ),
      mainPanel(
        h4("Movie Details"),
        tableOutput("movieDetails"),
        h4("Sentiment Analysis"),
        tableOutput("sentimentSummary"),
        h4("Keywords"),
        tableOutput("keywords")
      )
    )
  )
)


# Define Server
server <- function(input, output, session) {
  # Home Tab: Variable Descriptions
  output$variableDescriptions <- renderText({
    "Descriptions for variables used in this app"
  })
  
  variable_descriptions <- data.frame(
    Variable = c("Mgenres", "Tgenres", "NashSource", "NashProductionBudget", "sentiment_score"),
    Description = c(
      "Metacritic genres of the movie",
      "Tomatoes genres of the movie",
      "Source of the movie data",
      "Budget of the movie (in millions)",
      "Overall sentiment score of the reviews"
    )
  )
  
  output$varTable <- renderTable({
    variable_descriptions
  })
  
  # Predict Price Tab
  observeEvent(input$predict, {
    tryCatch({
      new_data <- data.frame(
        NashProductionBudget = input$NashProductionBudget,
        NashYear = input$NashYear,
        helpful = input$helpful,
        NashRunningTimeMinutes = input$NashRunningTimeMinutes,
        sentiment_score = input$sentiment_score
      )
      
      new_data$Mgenres <- factor("Unknown", levels = c("Unknown", "placeholder"))
      new_data$Tgenres <- factor("Unknown", levels = c("Unknown", "placeholder"))
      new_data$NashSource <- factor("Unknown", levels = c("Unknown", "placeholder"))
      new_data$NashProductionMethod <- factor("Unknown", levels = c("Unknown", "placeholder"))
      new_data$review_date <- as.Date("2000-01-01")
      new_data$NashMpaaRating <- factor("Unknown", levels = c("Unknown", "placeholder"))
      
      for (keyword in top_keywords) {
        new_data[[keyword]] <- 0
      }
      
      new_data$OtherKeywords <- 0
      
      new_data_encoded <- as.data.frame(predict(dummy_encoder, newdata = new_data))
      
      missing_columns <- setdiff(colnames(X_train_encoded), colnames(new_data_encoded))
      for (col in missing_columns) {
        new_data_encoded[[col]] <- 0
      }
      new_data_encoded <- new_data_encoded[, colnames(X_train_encoded), drop = FALSE]
      
      new_data_processed <- predict(preprocess_pipeline, as.data.frame(new_data_encoded))
      
      pred <- predict(lasso_model, newx = as.matrix(new_data_processed), s = "lambda.min")
      
      output$pricePrediction <- renderText({
        paste("Predicted Nash Domestic Box Office: $", round(pred, 2))
      })
    }, error = function(e) {
      output$pricePrediction <- renderText({
        paste("Error: ", e$message)
      })
    })
  })
  
  # Analyze Reviews Tab
  observeEvent(input$analyze, {
    filtered_reviews <- df %>%
      filter(MetacriticName == input$movie) %>%
      filter(if (!input$include_spoilers) spoiler_tag == 0 else TRUE)
    
    output$reviewTable <- renderTable({
      filtered_reviews %>%
        select(review_id, reviewer, review_summary, review_detail, rating, helpful)
    })
    
    output$ratingDistribution <- renderPlot({
      ggplot(filtered_reviews, aes(x = rating)) +
        geom_histogram(binwidth = 1, fill = "blue", color = "white") +
        theme_minimal() +
        labs(title = "Rating Distribution", x = "Rating", y = "Count")
    })
  })
  
  # Movie Info Tab
  observeEvent(input$get_movie_info, {
    req(input$movie_name)  # Ensure movie name is not empty
    
    # Filter dataset for the entered movie name
    movie_data <- df %>% filter(trimws(tolower(MetacriticName)) == trimws(tolower(input$movie_name)))
    
    if (nrow(movie_data) == 0) {
      # No data found for the movie
      output$movieDetails <- renderTable({
        data.frame(Message = "No data available for the entered movie name.")
      })
      output$sentimentSummary <- renderTable(NULL)
      output$keywords <- renderTable(NULL)
      return()
    }
    
    # Display basic movie details
    output$movieDetails <- renderTable({
      movie_data %>%
        select(MetacriticName, rating, NashRunningTimeMinutes, NashProductionBudget, NashYear) %>%
        distinct()
    })
    
    # Aggregate sentiment score
    sentiment_summary <- movie_data %>%
      summarise(
        Total_Positive = sum(sentiments$positive, na.rm = TRUE),
        Total_Negative = sum(sentiments$negative, na.rm = TRUE),
        Average_Sentiment_Score = mean(sentiment_score, na.rm = TRUE)
      )
    output$sentimentSummary <- renderTable({
      sentiment_summary
    })
    
    # Extract and summarize keywords from `review_summary`
    keyword_summary <- movie_data %>%
      select(review_summary) %>%
      unnest_tokens(word, review_summary) %>%  # Break text into individual words
      count(word, sort = TRUE) %>%
      filter(!word %in% stop_words$word) %>%  # Remove common stopwords like "the", "is", etc.
      slice_max(n = 10, order_by = n)         # Get the top 10 most frequent words
    
    output$keywords <- renderTable({
      keyword_summary %>% rename(Keyword = word, Count = n)
    })
  })
  
}

# Run the App
shinyApp(ui, server)
