# Level 0 (fixed target I): risk of SURE-chosen shrinkage relative to S, by dimension N.
# Spiked covariance Sigma = I + delta * e1 e1'; x-axis delta = ||Sigma - I||_F.
#
# Fast version. Every quantity depends on the data only through tr(S), tr(S^2) and s_11.
# Write X X' = (1 + delta) z z' + W, where z is stock 1's standardized returns and
# W is the T x T cross-product of the other N - 1 stocks. Neither z nor W depends on delta,
# so they are drawn once per N and reused for every delta (same draws: smooth curves).

library(tidyverse)
set.seed(2026)

T_days <- 40
dims   <- c(5, 10, 20, 50, 100, 200)
deltas <- seq(0, 40, by = 0.5)
n      <- 20000

curve_for_N <- function(N) {
  # four numbers per sample: tr W, ||W||^2, z'Wz, ||z||^2
  stats <- t(replicate(n, {
    z <- rnorm(T_days)
    W <- if (N - 1 < T_days) tcrossprod(matrix(rnorm(T_days * (N - 1)), T_days))
         else rWishart(1, N - 1, diag(T_days))[, , 1]
    c(sum(diag(W)), sum(W^2), sum(z * (W %*% z)), sum(z^2))
  }))
  trW <- stats[, 1]; W2 <- stats[, 2]; zWz <- stats[, 3]; zz <- stats[, 4]

  map_dbl(deltas, function(delta) {
    k    <- 1 + delta
    trS  <- (trW + k * zz) / T_days
    trS2 <- (W2 + 2 * k * zWz + k^2 * zz^2) / T_days^2
    s11  <- k * zz / T_days
    # SURE for the fixed target I: alpha-hat = sum Varhat(s_ij) / ||I - S||^2, kept in [0, 1]
    V <- (T_days * trS^2 + (T_days - 2) * trS2) / ((T_days + 2) * (T_days - 1))
    a <- pmin(pmax(V / (trS2 - 2 * trS + N), 0), 1)
    A <- 1 - a
    # losses ||a I + (1 - a) S - Sigma||^2 and ||S - Sigma||^2
    loss_sure <- A^2 * trS2 + 2 * A * (a * trS - trS - delta * s11) + (a - k)^2 + (N - 1) * (a - 1)^2
    loss_S    <- trS2 - 2 * (trS + delta * s11) + k^2 + N - 1
    sum(loss_sure) / sum(loss_S)
  })
}

Ratio <- map_dfr(dims, ~ tibble(N = .x, delta = deltas, Ratio = curve_for_N(.x)))

Ratio |>
  mutate(N = factor(N)) |>
  ggplot(aes(x = delta, y = Ratio, color = N)) +
    geom_line(linewidth = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray") +
    scale_color_manual(values = c("5" = "#c6dcf7", "10" = "#9ec5f4", "20" = "#5b9be6",
                                  "50" = "#2a78d6", "100" = "#1a56a8", "200" = "#0b2f5e")) +
    coord_cartesian(ylim = c(0, 1.02)) +
    labs(x = "Distance from the target, ‖Σ − I‖",
         y = "Risk over S", color = "N") +
    theme_minimal(base_size = 12)
ggsave("out/level0_risk_ratio_thesis.png", width = 7, height = 4, dpi = 200)
