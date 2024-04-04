library(mnormt)    # Multivariate Normal and t Distributions
library(ILAMM)     # Non-convex Regularized Robust Regression via I-LAMM Algorithm
library(doParallel)
library(foreach)
Rcpp::sourceCpp("./src/ILAMM.cpp") # Convoluted Rank Regression by ILAMM Algorithm and Metrics


pack <- function(X, y, beta.star, Sigma, file.name) {
    ## Sparse Convoluted Rank Regression by ILAMM Algorithm
    beta.scrr.lasso <- as.matrix(rr(X, y, lossType = "CRR", penalty = "Lasso"))
    beta.scrr.scad <- as.matrix(rr(X, y, lossType = "CRR", penalty = "SCAD"))
    beta.scrr.mcp <- as.matrix(rr(X, y, lossType = "CRR", penalty = "MCP"))
    
    # Rank Regression by ILAMM Algorithm
    beta.rr.lasso <- as.matrix(rr(X, y, lossType = "RR", penalty = "Lasso"))
    beta.rr.scad <- as.matrix(rr(X, y, lossType = "RR", penalty = "SCAD"))
    beta.rr.mcp <- as.matrix(rr(X, y, lossType = "RR", penalty = "MCP"))
    
    ## Adaptive Huber
    beta.huber.lasso <- ILAMM::cvNcvxHuberReg(X, y, penalty = "Lasso")$beta
    beta.huber.scad <- ILAMM::cvNcvxHuberReg(X, y, penalty = "SCAD")$beta
    beta.huber.mcp <- ILAMM::cvNcvxHuberReg(X, y, penalty = "MCP")$beta
    
    ## OLS
    beta.lasso <- ILAMM::cvNcvxReg(X, y, penalty = "Lasso")$beta
    beta.scad <- ILAMM::cvNcvxReg(X, y, penalty = "SCAD")$beta
    beta.mcp <- ILAMM::cvNcvxReg(X, y, penalty = "MCP")$beta
    
    metric <- c(
        # SCRR + Lasso metric
        l1_error(beta.scrr.lasso, beta.star),
        l2_error(beta.scrr.lasso, beta.star),
        ME(beta.scrr.lasso[-1], beta.star[-1], Sigma),
        FP(beta.scrr.lasso, beta.star),
        FN(beta.scrr.lasso, beta.star),
        MS(beta.scrr.lasso),
        
        # RR + Lasso metric
        l1_error(beta.rr.lasso, beta.star),
        l2_error(beta.rr.lasso, beta.star),
        ME(beta.rr.lasso[-1], beta.star[-1], Sigma),
        FP(beta.rr.lasso, beta.star),
        FN(beta.rr.lasso, beta.star),
        MS(beta.rr.lasso),
        
        # Adaptive Huber + Lasso metric
        l1_error(beta.huber.lasso, beta.star),
        l2_error(beta.huber.lasso, beta.star),
        ME(beta.huber.lasso[-1], beta.star[-1], Sigma),
        FP(beta.huber.lasso, beta.star),
        FN(beta.huber.lasso, beta.star),
        MS(beta.huber.lasso),
        
        # OLS + Lasso metric
        l1_error(beta.lasso, beta.star),
        l2_error(beta.lasso, beta.star),
        ME(beta.lasso[-1], beta.star[-1], Sigma),
        FP(beta.lasso, beta.star),
        FN(beta.lasso, beta.star),
        MS(beta.lasso),
        
        # SCRR + SCAD metric
        l1_error(beta.scrr.scad, beta.star),
        l2_error(beta.scrr.scad, beta.star),
        ME(beta.scrr.scad[-1], beta.star[-1], Sigma),
        FP(beta.scrr.scad, beta.star),
        FN(beta.scrr.scad, beta.star),
        MS(beta.scrr.scad),
        
        # RR + SCAD metric
        l1_error(beta.rr.scad, beta.star),
        l2_error(beta.rr.scad, beta.star),
        ME(beta.rr.scad[-1], beta.star[-1], Sigma),
        FP(beta.rr.scad, beta.star),
        FN(beta.rr.scad, beta.star),
        MS(beta.rr.scad),
        
        # Adaptive Huber + SCAD metric
        l1_error(beta.huber.scad, beta.star),
        l2_error(beta.huber.scad, beta.star),
        ME(beta.huber.scad[-1], beta.star[-1], Sigma),
        FP(beta.huber.scad, beta.star),
        FN(beta.huber.scad, beta.star),
        MS(beta.huber.scad),
        
        # OLS + SCAD metric
        l1_error(beta.scad, beta.star),
        l2_error(beta.scad, beta.star),
        ME(beta.scad[-1], beta.star[-1], Sigma),
        FP(beta.scad, beta.star),
        FN(beta.scad, beta.star),
        MS(beta.scad),
        
        # SCRR + MCP metric
        l1_error(beta.scrr.mcp, beta.star),
        l2_error(beta.scrr.mcp, beta.star),
        ME(beta.scrr.mcp[-1], beta.star[-1], Sigma),
        FP(beta.scrr.mcp, beta.star),
        FN(beta.scrr.mcp, beta.star),
        MS(beta.scrr.mcp),
        
        # RR + MCP metric
        l1_error(beta.rr.mcp, beta.star),
        l2_error(beta.rr.mcp, beta.star),
        ME(beta.rr.mcp[-1], beta.star[-1], Sigma),
        FP(beta.rr.mcp, beta.star),
        FN(beta.rr.mcp, beta.star),
        MS(beta.rr.mcp),
        
        # Adaptive Huber + MCP metric
        l1_error(beta.huber.mcp, beta.star),
        l2_error(beta.huber.mcp, beta.star),
        ME(beta.huber.mcp[-1], beta.star[-1], Sigma),
        FP(beta.huber.mcp, beta.star),
        FN(beta.huber.mcp, beta.star),
        MS(beta.huber.mcp),
        
        # OLS + MCP metric
        l1_error(beta.mcp, beta.star),
        l2_error(beta.mcp, beta.star),
        ME(beta.mcp[-1], beta.star[-1], Sigma),
        FP(beta.mcp, beta.star),
        FN(beta.mcp, beta.star),
        MS(beta.mcp)
    )
    
    ## Save the metric appending a csv file
    write.table(
        t(metric),
        # file = "TFER_AHuber_OLS.csv",
        file = file.name,
        append = TRUE,
        sep = ",",
        row.names = FALSE,
        col.names = FALSE
    )
    
    # return(metric)
}

n <- 50
d <- 100

beta.star <- as.matrix(c(0, 5,-2, 0, 0, 3, rep(0, d - 5))) # length = d+1
set.seed(1)
## X ~ N(0, I)
Sigma <- diag(d)
X <- rmnorm(n, mean = rep(0, d), varcov = Sigma)

num_cores <- detectCores() - 4
cl <- makeCluster(num_cores)
registerDoParallel(cl)
foreach(i = 1:100, .combine = rbind) %dopar% function(i) {
    y <- cbind(1, X) %*% beta.star + matrix(rnorm(n), ncol = 1)
    return(pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_1_N0I_N01.csv"))
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rt(n, df = 2))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_2_N0I_t2.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rlnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_3_N0I_logNormal.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rcauchy(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_4_N0I_Cauchy.csv")
}

set.seed(1)
# X ~ N(0, 0.9)
Sigma <- mapply(\(x, y) 0.9 ^ (abs(x - 1:y)), 1:d, d)
X <- rmnorm(n, mean = rep(0, d), varcov = Sigma)
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_5_N09_N01.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rt(n, df = 2))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_6_N09_t2.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rlnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_7_N09_logNormal.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rcauchy(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_8_N09_Cauchy.csv")
}

set.seed(1)
# X ~ N(0, 0.5)
Sigma <- mapply(\(x, y) 0.5 ^ (abs(x - 1:y)), 1:d, d)
X <- rmnorm(n, mean = rep(0, d), varcov = Sigma)
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_9_N05_N01.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rt(n, df = 2))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_10_N05_t2.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rlnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_11_N05_logNormal.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rcauchy(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_12_N05_Cauchy.csv")
}

set.seed(1)
# X ~ N(0, 0.1)
Sigma <- mapply(\(x, y) 0.1 ^ (abs(x - 1:y)), 1:d, d)
X <- rmnorm(n, mean = rep(0, d), varcov = Sigma)
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_13_N01_N01.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rt(n, df = 2))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_14_N01_t2.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rlnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_15_N01_logNormal.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rcauchy(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_16_N01_Cauchy.csv")
}

set.seed(1)
# X ~ N(0, 0.5)
Sigma <- mapply(\(x, y) ifelse(x == 1:y, 1, 0.5), 1:d, d)
X <- rmnorm(n, mean = rep(0, d), varcov = Sigma)
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_17_N0_5_N01.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rt(n, df = 2))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_18_N0_5_t2.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rlnorm(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_19_N0_5_logNormal.csv")
}
foreach(i = 1:100) %dopar% {
    y <- cbind(1, X) %*% beta.star + as.matrix(rcauchy(n))
    pack(X, y, beta.star, Sigma, "newSCRR_RR_AHuber_OLS_20_N0_5_Cauchy.csv")
}

stopCluster(cl)
