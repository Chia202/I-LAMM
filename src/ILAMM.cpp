# include <RcppArmadillo.h>
# include <cmath>
# include <string>
# include <armadillo>

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

arma::vec softThresh(const arma::vec& x, const arma::vec& lambda) {
    return arma::sign(x) % arma::max(arma::abs(x) - lambda, arma::zeros(x.size()));
}

double crrLoss(const arma::mat& X, const arma::vec& y, const arma::vec& beta, const double& h = 1) {
    arma::vec res = y - X * beta;
    arma::vec loss = arma::zeros<arma::vec>(res.size());
    
    // Calculate loss for all data points simultaneously
    loss = arma::conv_to<arma::vec>::from(arma::abs(res) <= h) % (3.0 / 4 * arma::pow(res, 2) / h - arma::pow(res, 4) / 8.0 / pow(h, 3) + 3.0 / 8 * h)
        + arma::conv_to<arma::vec>::from(arma::abs(res) > h) % arma::abs(res);
    
    // Return the average loss
    return arma::mean(loss);
}

arma::vec crrLossGrad(const arma::mat& X, const arma::vec& y, const arma::vec& beta, const double& h = 1) {
    arma::vec res = y - X * beta;
    
    // Calculate the gradient for all data points simultaneously
    arma::uvec idxLessH = arma::find(arma::abs(res) <= h);
    arma::uvec idxGreaterH = arma::find(arma::abs(res) > h);
    
    res.rows(idxLessH) = (3.0 / 2 * res.rows(idxLessH) / h - arma::pow(res.rows(idxLessH), 3) / 2.0 / pow(h, 3));
    res.rows(idxGreaterH) = arma::sign(res.rows(idxGreaterH));
    
    return(-X.t() * res / res.size());
}

double rrLoss(const arma::mat& X, const arma::vec& y, const arma::vec& beta) {
    return(arma::mean(arma::abs(y - X * beta)));
}

arma::vec rrLossGrad(const arma::mat& X, const arma::vec& y, const arma::vec& beta) {
    return(-X.t() * arma::sign(y - X * beta) / y.size());
}

double logisticLoss(const arma::mat& X, const arma::vec& y, const arma::vec& beta) {
    arma::vec p = 1.0 / (1.0 + exp(-X * beta)); // Compute predicted probabilities
    
    // std::cout << "p = " << p << std::endl;
    double loss = -dot(y, log(p)) - dot(1 - y, log(1 - p)); // Compute logistic loss
    
    // std::cout << "loss = " << loss << std::endl;
    return loss;
}

arma::vec logisticLossGrad(const arma::mat& X, const arma::vec& y, const arma::vec& beta) {
    arma::vec p = 1.0 / (1.0 + exp(-X * beta)); // Compute predicted probabilities
    
    arma::vec grad = X.t() * (p - y); // Compute gradient
    
    return grad;
}

double loss(const arma::mat& X, const arma::vec& y, 
            const arma::vec& beta, const std::string& lossType, const double& h = 1) {
    if (lossType == "CRR") {
        return crrLoss(X, y, beta, h);
    } else if (lossType == "RR") {
        return rrLoss(X, y, beta);
    } else if (lossType == "logistic") {
        return logisticLoss(X, y, beta);
    } else {
        return 0;
    }
}

arma::vec lossGrad(const arma::mat& X, const arma::vec& y, 
                   const arma::vec& beta, const std::string& lossType, const double& h = 1) {
    if (lossType == "CRR") {
        return crrLossGrad(X, y, beta, h);
    } else if (lossType == "RR") {
        return rrLossGrad(X, y, beta);
    } else if (lossType == "logistic") {
        return logisticLossGrad(X, y, beta);
    } else {
        return 0;
    }
}

double cmptPsi(const arma::mat& X, const arma::vec& y, const arma::vec& betaNew,
               const arma::vec& beta, const double phi, const double& h = 1) {
    arma::vec diff = betaNew - beta;
    return dot(diff, crrLossGrad(X, y, beta, h)) + phi * norm(diff, 2);
}

// Compute the weight of the penalty term
arma::vec cmptLambda(const arma::vec& beta, const double& lambda, const std::string& penalty) {
    arma::vec rst = arma::zeros(beta.size());
    if (penalty == "Lasso") {
        rst = lambda * arma::ones(beta.size());
        rst(0) = 0;
    } else if (penalty == "SCAD") {
        double a = 3.7;
        double abBeta;
        for (int i = 1; i < (int)beta.size(); i++) {
            abBeta = std::abs(beta(i));
            if (abBeta <= lambda) {
                rst(i) = lambda;
            } else if (abBeta <= a * lambda) {
                rst(i) = (a * lambda - abBeta) / (a - 1);
            }
        }
    } else if (penalty == "MCP") {
        double a = 3;
        double abBeta;
        for (int i = 1; i < (int)beta.size(); i++) {
            abBeta = std::abs(beta(i));
            if (abBeta <= a * lambda) {
                rst(i) = lambda - abBeta / a;
            }
        }
    }
    return rst;
}

double cmptHBIC(const arma::mat& X, const arma::vec& y, const int& n,
                const arma::vec& beta, const std::string& lossType = "RR",
                const double& h = 1) {
    double p = beta.size();
    double rst = loss(X, y, beta, lossType, h);
    return log(rst) + log(log(n)) / n * log(p) * arma::sum(beta != 0);
}

//[[Rcpp::export]]
Rcpp::List rrLambda(const arma::mat& X, const arma::vec& y, const double& n,
                     const double& lambda = 1.0, const std::string& lossType = "RR",
                     const std::string& penalty = "Lasso") {
    double phi   = 0.001;
    double eps   = 1e-5;
    double gamma = 2.0;
    double h     = 1.0;
    
    double p     = X.n_cols;
    int iterMax  = 1000;
    
    arma::vec beta = arma::zeros(p);
    arma::vec betaNew = arma::ones(p);
    
    // Contraction
    for (int iter = 0; iter < iterMax; iter++) {
        for (int j = 0; j < iterMax; j++) {
            betaNew = softThresh(beta - lossGrad(X, y, beta, lossType, h) / phi, cmptLambda(beta, lambda, "Lasso") / phi);
            if (cmptPsi(X, y, betaNew, beta, phi, h) < 0) {
                phi *= gamma;
            } else {
                break;
            }
        }
        if (norm(betaNew - beta, 2) < eps) {
            beta = betaNew;
            break;
        }
        beta = betaNew;
    }
    
    // Tightening
    for (int iter = 0; iter < iterMax; iter++) {
        for (int j = 0; j < iterMax; j++) {
            betaNew = softThresh(beta - lossGrad(X, y, beta, lossType, h) / phi, cmptLambda(beta, lambda, penalty) / phi);
            if (cmptPsi(X, y, betaNew, beta, phi, h) < 0) {
                phi *= gamma;
            } else {
                break;
            }
        }
        if (norm(betaNew - beta, 2) < eps) {
            beta = betaNew;
            break;
        }
        beta = betaNew;
    }
    
    return(Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("HBIC") = cmptHBIC(X, y, n, beta, lossType, h),
                              Rcpp::Named("phi")  = phi));
}

//[[Rcpp::export]]
arma::vec rr(const arma::mat& X, const arma::vec& y, const std::string& lossType = "RR", const std::string& penalty = "Lasso") {
    int n = X.n_rows;
    int p = X.n_cols;
    
    arma::mat Xnew = arma::zeros(n * (n-1), p);
    arma::vec ynew = arma::zeros(n * (n-1));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                Xnew.row(i * (n-1) + j) = X.row(i) - X.row(j);
                ynew(i * (n-1) + j) = y(i) - y(j);
            }
        }
    }
    
    std::cout << "dim of Xnew: " << Xnew.n_rows << " x " << Xnew.n_cols << std::endl;
    
    double bestLambda = 0;
    double minHBIC = std::numeric_limits<double>::infinity();
    
    for (double lambda = 0.01; lambda < 1; lambda += 0.01) {
        Rcpp::List tmp = rrLambda(Xnew, ynew, n, lambda, lossType, penalty);
        if (Rcpp::as<double>(tmp["HBIC"]) < minHBIC) {
            minHBIC = tmp["HBIC"];
            bestLambda = lambda;
            std::cout << "lambda = " << lambda << ", HBIC = " << minHBIC << std::endl;
        }
    }
    
    arma::vec beta = Rcpp::as<arma::vec>(rrLambda(Xnew, ynew, n, bestLambda, lossType, penalty)["beta"]);
    double mean_error = arma::mean(y - X * beta);
    std::cout << "Mean error: " << mean_error << std::endl;
    beta.insert_rows(0, 1);
    beta[0] = mean_error;
    return beta;
}

//[[Rcpp::export]]
Rcpp::List glmLambda(const arma::mat& X, const arma::vec& y, const double& lambda,
                     const std::string& lossType = "logistic", const std::string& penalty = "Lasso") {
    double phi   = 0.001;
    double eps   = 1e-5;
    double gamma = 2.0;
    double h     = 1.0;
    
    double p     = X.n_cols;
    int iterMax  = 1000;
    
    arma::vec beta = arma::zeros(p);
    arma::vec betaNew = arma::ones(p);
    
    // Contraction
    for (int iter = 0; iter < iterMax; iter++) {
        for (int j = 0; j < iterMax; j++) {
            betaNew = softThresh(beta - lossGrad(X, y, beta, lossType, h) / phi, cmptLambda(beta, lambda, "Lasso") / phi);
            // std::cout << "betaNew = " << betaNew << std::endl;
            if (cmptPsi(X, y, betaNew, beta, phi, h) < 0) {
                phi *= gamma;
            } else {
                break;
            }
        }
        if (norm(betaNew - beta, 2) < eps) {
            beta = betaNew;
            break;
        }
        beta = betaNew;
    }
    
    // Tightening
    for (int iter = 0; iter < iterMax; iter++) {
        for (int j = 0; j < iterMax; j++) {
            betaNew = softThresh(beta - lossGrad(X, y, beta, lossType, h) / phi, cmptLambda(beta, lambda, penalty) / phi);
            if (cmptPsi(X, y, betaNew, beta, phi, h) < 0) {
                phi *= gamma;
            } else {
                break;
            }
        }
        if (norm(betaNew - beta, 2) < eps) {
            beta = betaNew;
            break;
        }
        beta = betaNew;
    }
    
    return(Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("HBIC") = cmptHBIC(X, y, X.n_rows, beta, lossType, h),
                              Rcpp::Named("phi")  = phi));
}

//[[Rcpp::export]]
arma::vec glm(arma::mat X, arma::vec y, std::string lossType = "logistic", std::string penalty = "Lasso") {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
    
    double bestLambda = 0;
    double minHBIC = std::numeric_limits<double>::infinity();
    
    for (double lambda = 0.01; lambda < 3; lambda += 0.01) {
        Rcpp::List tmp = glmLambda(X, y, lambda, lossType, penalty);
        if (Rcpp::as<double>(tmp["HBIC"]) < minHBIC) {
            minHBIC = tmp["HBIC"];
            bestLambda = lambda;
            std::cout << "lambda = " << lambda << ", HBIC = " << minHBIC << std::endl;
        }
    }
    
    return(Rcpp::as<arma::vec>(glmLambda(X, y, bestLambda)["beta"]));
}
