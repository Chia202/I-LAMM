library(ILAMM)
library(glmnet)
library(pROC)
Rcpp::sourceCpp("./src/ILAMM.cpp") # Convoluted Rank Regression by ILAMM Algorithm and Metrics

######################## Linear Rgression    ########################
load("../Data/eyedata.rda")

cmptMSE <- function(beta, X, y) {
    mean((cbind(1, X) %*% beta - y)^2)
}

n <- nrow(genes)
for (i in 1:100) {
    index <- sample(1:n, n/2) 
    
    X <- genes[index, ]
    y <- trim32[index]
    
    ## Sparse Convoluted Rank Regression by ILAMM Algorithm
    beta.scrr.lasso <- as.matrix(rr(X, y, lossType = "CRR", penalty = "Lasso"))
    beta.scrr.scad  <- as.matrix(rr(X, y, lossType = "CRR", penalty = "SCAD"))
    beta.scrr.mcp   <- as.matrix(rr(X, y, lossType = "CRR", penalty = "MCP"))
    
    # Rank Regression by ILAMM Algorithm
    beta.rr.lasso <- as.matrix(rr(X, y, lossType = "RR", penalty = "Lasso"))
    beta.rr.scad  <- as.matrix(rr(X, y, lossType = "RR", penalty = "SCAD"))
    beta.rr.mcp   <- as.matrix(rr(X, y, lossType = "RR", penalty = "MCP"))
    
    ## Adaptive Huber
    beta.huber.lasso <- ILAMM::cvNcvxHuberReg(X, y, penalty = "Lasso")$beta
    beta.huber.scad  <- ILAMM::cvNcvxHuberReg(X, y, penalty = "SCAD")$beta
    beta.huber.mcp   <- ILAMM::cvNcvxHuberReg(X, y, penalty = "MCP")$beta
    
    ## OLS
    beta.lasso <- ILAMM::cvNcvxReg(X, y, penalty = "Lasso")$beta
    beta.scad  <- ILAMM::cvNcvxReg(X, y, penalty = "SCAD")$beta
    beta.mcp   <- ILAMM::cvNcvxReg(X, y, penalty = "MCP")$beta
    
    X <- genes[-index, ]
    y <- trim32[-index]
    
    metric <- c(
        cmptMSE(beta.scrr.lasso, X, y), cmptMSE(beta.scrr.scad, X, y), cmptMSE(beta.scrr.mcp, X, y),
        cmptMSE(beta.rr.lasso, X, y), cmptMSE(beta.rr.scad, X, y), cmptMSE(beta.rr.mcp, X, y),
        cmptMSE(beta.huber.lasso, X, y), cmptMSE(beta.huber.scad, X, y), cmptMSE(beta.huber.mcp, X, y),
        cmptMSE(beta.lasso, X, y), cmptMSE(beta.scad, X, y), cmptMSE(beta.mcp, X, y)
    )
    
    write.table(metric, file = "GenesBeta.csv", append = TRUE, col.names = FALSE, row.names = FALSE)
}

######################## Logistic Regression ########################
library(pROC)
load("../Data/golub.RData")
golub$x <- scale(golub$x)
golub$y <- ifelse(golub$y == 'ALL', 1, 0)
pred.p <- matrix(NA, nrow = nrow(golub$x), ncol = 4)

for (i in 1:nrow(golub$x)) {
    X_train <- golub$x[-i, ]
    y_train <- golub$y[-i]
    
    golub.glm <- coef(glmnet::cv.glmnet(X_train, y_train, family = "binomial", alpha = 1))
    golub.lasso <- glm(X_train, y_train, 'logistic', penalty = "Lasso")
    golub.scad <- glm(X_train, y_train, 'logistic', penalty = "SCAD")
    golub.mcp <- glm(X_train, y_train, 'logistic', penalty = "MCP")
    
    pred.p[i, 1] <- 1 / (1 + exp(-c(1, golub$x[i, ]) %*% golub.glm))
    pred.p[i, 2] <- 1 / (1 + exp(-c(1, golub$x[i, ]) %*% golub.lasso))
    pred.p[i, 3] <- 1 / (1 + exp(-c(1, golub$x[i, ]) %*% golub.scad))
    pred.p[i, 4] <- 1 / (1 + exp(-c(1, golub$x[i, ]) %*% golub.mcp))
}

write.table(pred.p, file = "pred_p.csv", col.names = FALSE, row.names = FALSE)
