
# LOADING DATA
titanic_x_train <- read.csv(('data/titanic_df_X_train.csv'))
titanic_x_test  <- read.csv(('data/titanic_df_X_test.csv'))
titanic_y_train <- read.csv(('data/titanic_df_y_train.csv'))
titanic_y_test  <- read.csv(('data/titanic_df_y_test.csv'))
bank_x_train    <- read.csv(('data/bank_df_X_train.csv'))
bank_x_test     <- read.csv(('data/bank_df_X_test.csv'))
bank_y_train    <- read.csv(('data/bank_df_y_train.csv'))
bank_y_test     <- read.csv(('data/bank_df_y_test.csv'))

# MERGING X AND Y DATAFRAMES
titanic_train <- cbind(titanic_x_train, titanic_y_train)
titanic_test <- cbind(titanic_x_test, titanic_y_test)
bank_train    <- cbind(bank_x_train, bank_y_train)
bank_test    <- cbind(bank_x_test, bank_y_test)

# FITTING MODELS
library(faraway)
# glm model from faraway uses IRLS
model_titanic <- glm(unlist(Survived)~ ., family=binomial, data=titanic_train)
model_bank <- glm(unlist(class)~ ., family=binomial, data=bank_train)

summary(model_titanic)
summary(model_bank)

# EVALUATING MODELS
library(caret)

pred_titanic <- predict(model_titanic, titanic_x_test, type="response")
actual_titanic <- titanic_y_test
pred_bank <- predict(model_bank, bank_x_test, type="response")
actual_bank <- bank_y_test
pred_titanic <-as.integer(pred_titanic > 0.5)
pred_bank <- as.integer(pred_bank > 0.5)

cm_titanic <- confusionMatrix(as.factor(pred_titanic), as.factor(actual_titanic$Survived), mode = "everything", positive="1")
cm_bank <- confusionMatrix(as.factor(pred_bank), as.factor(actual_bank$class), mode = "everything", positive="1")

metrics_titanic <- c(cm_titanic$overall[1], cm_titanic$byClass[c(5, 6, 7, 11)])
metrics_bank <- c(cm_bank$overall[1], cm_bank$byClass[c(5, 6, 7, 11)])

# CROSS VALIDATION

titanic <- rbind(titanic_train, titanic_test)
bank <- rbind(bank_train, bank_test)

# TITANIC DATASET
acc_t   <- c()
acc_b_t <- c()
rec_t   <- c()
prec_t  <- c()
f1_t    <- c()
N = 50
for (i in 1:N){
  for (j in 1:10){
    idx <- createDataPartition(titanic$Survived, p = .9, list = FALSE)
    train <- titanic[ idx,]
    test  <- titanic[-idx,]
    model_titanic <- glm(unlist(Survived)~ ., family=binomial, data=train)
    pred_titanic <- predict(model_titanic, subset(test, select=-Survived), type="response")
    pred_titanic <-as.integer(pred_titanic > 0.5)
    actual_titanic <- test$Survived
    cm_titanic <- confusionMatrix(as.factor(pred_titanic), as.factor(actual_titanic), mode = "everything", positive="1")
    acc_t   <- c(acc_t, cm_titanic$overall[1])
    acc_b_t <- c(acc_b_t, cm_titanic$byClass[11])
    rec_t   <- c(rec_t, cm_titanic$byClass[6])
    prec_t  <- c(prec_t, cm_titanic$byClass[5])
    f1_t    <- c(f1_t, cm_titanic$byClass[7])
  }
}

# BANKNOTE DATASET
acc_b   <- c()
acc_b_b <- c()
rec_b   <- c()
prec_b  <- c()
f1_b    <- c()
N = 50
for (i in 1:N){
  for (j in 1:10){
    idx <- createDataPartition(bank$class, p = .9, list = FALSE)
    train <- bank[ idx,]
    test  <- bank[-idx,]
    model_bank <- glm(unlist(class)~ ., family=binomial, data=train)
    pred_bank <- predict(model_bank, subset(test, select=-class), type="response")
    pred_bank <-as.integer(pred_bank > 0.5)
    actual_bank <- test$class
    cm_bank <- confusionMatrix(as.factor(pred_bank), as.factor(actual_bank), mode = "everything", positive="1")
    acc_b   <- c(acc_b, cm_bank$overall[1])
    acc_b_b <- c(acc_b_b, cm_bank$byClass[11])
    rec_b   <- c(rec_b, cm_bank$byClass[6])
    prec_b  <- c(prec_b, cm_bank$byClass[5])
    f1_b    <- c(f1_b, cm_bank$byClass[7])
  }
}


# WRITING RESULTS


write(metrics_titanic, "metrics_titanic.txt")
write(metrics_bank, "metrics_bank.txt")
write(acc_t, "acc_t.txt")
write(acc_b_t, "acc_b_t.txt")
write(rec_t, "rec_t.txt")
write(prec_t, "prec_t.txt")
write(f1_t, "f1_t.txt") 
write(acc_b, "acc_b.txt")
write(acc_b_b, "acc_b_b.txt")
write(rec_b, "rec_b.txt")  
write(prec_b, "prec_b.txt")
write(f1_b, "f1_b.txt")   





















