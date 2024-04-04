# I-LAMM

This repo implements iterative local adaptive majorize-minimization algorithm proposed by [Fan et al](https://doi.org/10.1214/17-AOS1568) based on my understanding.

## Functions

- `rr`: Rank regression related function penalized by Lasso, SCAD, MCP, e.g. smooth convoluted rank regression by [Zhou et al](https://doi.org/10.1080/01621459.2023.2202433) and rank regression by [Wang et al](https://doi.org/10.1080/01621459.2020.1840989).  
- `glm`: Generalized linear regression penalized by Lasso, SCAD, MCP.

## Usage

```r
n <- 100
p <- 200
beta    <- as.matrix(c(5, 2, 0, 0, -3, rep(0, p - 5)), nrow = p)
X       <- matrix(rnorm(n*p), nrow = n)
epsilon <- matrix(rnorm(n),nrow = n)
y       <- X %*% beta + epsilon

## SCRR
scrr.lasso <- rr(X, y, "CRR", "Lasso")
scrr.scad  <- rr(X, y, "CRR", "SCAD")
scrr.mcp   <- rr(X, y, "CRR", "MCP")

## Rank
rr.lasso <- rr(X, y, "RR", "Lasso")
rr.scad  <- rr(X, y, "RR", "SCAD")
rr.mcp   <- rr(X, y, "RR", "MCP")

# Convert y into binary variable
y <- (y > 0)
## Logistic regression
lr.lasso <- glm(X, y, penalty = "Lasso")
lr.scad  <- glm(X, y, penalty = "SCAD")
lr.mcp   <- glm(X, y, penalty = "MCP")
```

## Reference

[1]. FAN J, LIU H, SUN Q, et al. I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error[J]. The Annals of Statistics, 2018, 46(2): 814-841.

[2]. WANG L, PENG B, BRADIC J, et al. A Tuning-free Robust and Eﬀicient Approach to High-dimensional Regression[J]. Journal of the American Statistical Association, 2020, 115(532): 1700-1714.

[3]. ZHOU L, WANG B, ZOU H. Sparse Convoluted Rank Regression in High Dimensions[J]. Journal of the American Statistical Association, 2023, 0(0): 1-13.
