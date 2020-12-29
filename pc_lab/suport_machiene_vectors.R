# Outlook to Machine Learning: Support Vector Machines
# 16.12.2020


#Linearly separable data example 

#Create example data ----

# Define random seed 
set.seed(5555)

x = matrix(rnorm(40), 20, 2)
y = rep(c(-1, 1), c(10, 10))
x[y == 1,] = x[y == 1,] + 1

plot(x, col = y + 3, pch = 19)



# Train a SVM model ----

library(e1071)
dat = data.frame(x, y = as.factor(y))
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)
print(svmfit)

# SVM classification plot ----
plot(svmfit, dat, pch = 19)

# Visualize support and decision boundary vectors ----
## The following is only for illustration of the decision boundaries and typically not included in actual SVM analysis projects.

# Extract linear coefficients of the SVM model
beta = drop(t(svmfit$coefs)%*%x[svmfit$index,])
beta0 = svmfit$rho

# Create a grid showing the domains of class -1 vs. class 1 
make.grid = function(x, n = 75) {
  grange = apply(x, 2, range)
  x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}

xgrid = make.grid(x)
ygrid = predict(svmfit, xgrid)

# Plot grid, data points and vectors
plot(xgrid, col = c("red", "blue")[as.numeric(ygrid)], pch = 20, cex = .2)
points(x, col = y + 3, pch = 19)
points(x[svmfit$index,], pch = 5, cex = 2)
abline(beta0 / beta[2], -beta[1] / beta[2])
abline((beta0 - 1) / beta[2], -beta[1] / beta[2], lty = 2)
abline((beta0 + 1) / beta[2], -beta[1] / beta[2], lty = 2)


# Kernals ----

# Non-linear example data
# Let’s look at a data set that is not linearly separable …

# Clean up a bit ...
rm(x, y)

# Load environment from RData file
load(file = "~/Documents/humbolt/quantitative_methods/pc_lab/data/ESL.mixture.rda")
names(ESL.mixture)
## [1] "x"        "y"        "xnew"     "prob"     "marginal" "px1"      "px2"      "means"
attach(ESL.mixture)

p <- plot(x, col = y + 1)


# Train a non-linear SVM classifier ----

# Convert the response variable to factor
dat = data.frame(y = factor(y), x)

# Train a SVM model. Note the kernel being set to 'radial'.
fit = svm(factor(y) ~ ., data = dat, scale = FALSE, kernel = "radial", cost = 5)


# Visualize support and decision boundary vectors ----

# Create grid background
xgrid = expand.grid(X1 = px1, X2 = px2)
ygrid = predict(fit, xgrid)

# Plot grid and data points
plot(xgrid, col = as.numeric(ygrid), pch = 20, cex = .2)
points(x, col = y + 1, pch = 19)

# Vizualize decision boundary
func = predict(fit, xgrid, decision.values = TRUE)
func = attributes(func)$decision
contour(px1, px2, matrix(func, 69, 99), level = 0, add = TRUE)



# SVM with multidimensional data --------------------------------------------------------------------------------------

# Multidimensional example ----

# Measurements of sonar reflection characteristics for rock (‘R’) and metal (‘M’). Columns represent different angles at which the sonar chirps hit the objects.

sonar.data <- read.csv("~/Documents/humbolt/quantitative_methods/pc_lab/data/sonar.all-data", stringsAsFactors = T)
head(sonar.data, 1)

# Create test and train subsets ----

sonar.n <- nrow(sonar.data)

# Use 20 % of the total data as test data
test_portion <- 0.2
test_size <- round(sonar.n * test_portion)
test_n <- round(sonar.n * test_portion)
train_n <- round(sonar.n * (1 - test_portion))

test_samples <- sample(nrow(sonar.data), test_n)

test_df <- sonar.data[test_samples,]
train_df <- sonar.data[-test_samples,]

train_X <- subset(train_df, select = -R)
train_y <- train_df$R

test_X  <- subset(test_df, select = -R)
test_y  <- test_df$R


# Setup initial SVM model ----

model <- svm(train_X, train_y)
print(model)

summary(model)

# Check initial SVM model’s performance
pred <- predict(model, train_X)
pred_test <- predict(model, test_X)

# Confusion matrix table for train data
# (how well does the model perform describing the data it was trained on?)
table(pred, train_y)

# Confusion matrix table for test data
# (how well does the model perform classifying unknown data?)
table(pred_test, test_y)

n_correct <- length(pred_test[pred_test == test_y])
n_test <- length(pred_test)

paste('Correct [%]: ', n_correct / n_test * 100)


# Hyperparameter Tuning ---------------------------------------------------------------------------------------------

ranges_conf <- list(cost=10^(-1:3), gamma=c(.001,.002,.005,.01,.02,.05,.1,.2,.5,1,2))

svm_tune <- tune(svm, train.x=train_X, train.y=train_y, ranges=ranges_conf)
print(svm_tune)

svm_tune$performances
svm_tune$performances %>% arrange(error)

# best model

best_model <- svm_tune$best.model
print(best_model)


# Check the tuned model ----

pred_best <- predict(best_model, train_X)
# Confusion matrix table for train data using tuned model
table(pred_best, train_y)

pred_best_test <- predict(best_model, test_X)
# Confusion matrix table for test data using tuned model
table(pred_best_test, test_y)

n_correct <- length(pred_best_test[pred_best_test == test_y])
n_test <- length(pred_best_test)

paste('Correct [%]: ', n_correct / n_test * 100)
