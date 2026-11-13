# Why the Level 0 curve has a hump: spread of alpha-hat and SURE vs oracle risk ratio along the spike path
set.seed(1); Tn <- 40; n <- 20000
run <- function(N, deltas) {
  st <- t(replicate(n, { z <- rnorm(Tn); W <- tcrossprod(matrix(rnorm(Tn*(N-1)), Tn))
    c(sum(diag(W)), sum(W^2), sum(z*(W%*%z)), sum(z^2)) }))
  trW <- st[,1]; W2 <- st[,2]; zWz <- st[,3]; zz <- st[,4]
  for (d in deltas) { k <- 1+d
    trS <- (trW+k*zz)/Tn; trS2 <- (W2+2*k*zWz+k^2*zz^2)/Tn^2; s11 <- k*zz/Tn
    V <- (Tn*trS^2+(Tn-2)*trS2)/((Tn+2)*(Tn-1)); a <- pmin(pmax(V/(trS2-2*trS+N),0),1)
    loss <- function(a) { A <- 1-a; A^2*trS2 + 2*A*(a*trS-trS-d*s11) + (a-k)^2 + (N-1)*(a-1)^2 }
    lS <- trS2-2*(trS+d*s11)+k^2+N-1
    Vt <- ((N+d)^2+N+2*d+d^2)/Tn; astar <- Vt/(Vt+d^2)
    cat(sprintf("N=%3d dist=%6.1f  best alpha=%.3f  alpha-hat mean=%.3f sd=%.3f  ratio oracle=%.3f  ratio SURE=%.3f  gap=%.3f\n",
        N, d, astar, mean(a), sd(a), mean(loss(astar))/mean(lS), mean(loss(a))/mean(lS), (mean(loss(a))-mean(loss(astar)))/mean(lS)))
  }
}
run(5,  c(0,1,2,4,6.5,10,20,40,100,400))
run(20, c(10,20,40,80,150,400))
run(50, c(40,80,150,300,800))
