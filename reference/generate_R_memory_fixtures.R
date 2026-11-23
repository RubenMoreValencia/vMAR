# Run from the C vMAR root, with R vMAR.Core >= 0.0.8 loaded.
if (!exists("vmar_neuron_loss_gradient", mode = "function"))
  library(vMAR.Core)

x <- matrix(c(1, 0.5, -0.5, 1.5, 0.7, -1), nrow = 3, byrow = TRUE)
target <- c(0.25, -0.3, 0.7)
theta0 <- c(0.2, -0.4, 0.15)
cases <- list()

for (activation in c("tanh", "sigmoid")) {
  for (method in c("euler", "rk4")) {
    for (coupling in c(0, 0.2)) {
      for (steps in c(1L, 12L)) {
        eta <- 0.3; tau <- 0.6; h <- 0.1
        field <- function(time, state) {
          theta <- state[1:3]; memory <- state[4:6]
          gradient <- vmar_neuron_loss_gradient(x, target, theta,
                                               activation)$gradient
          c(-eta * (gradient + coupling * memory),
            (gradient - memory) / tau)
        }
        advance <- if (method == "euler") vmar_euler_step else vmar_rk4_step
        state <- c(theta0, c(0,0,0))
        for (k in seq_len(steps))
          state <- advance(field, (k-1L)*h, state, h)
        cases[[length(cases)+1L]] <- data.frame(
          id=paste(activation,method,paste0("c",coupling),
                   paste0("n",steps),sep="_"),
          activation=activation,method=method,coupling=coupling,
          eta=eta,tau=tau,step=h,steps=steps,
          theta1=state[1],theta2=state[2],theta3=state[3],
          memory1=state[4],memory2=state[5],memory3=state[6])
      }
    }
  }
}
utils::write.csv(do.call(rbind,cases), "reference/vmar_memory_parity.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.3 memory fixtures generated: 16 cases\n")

# Separate exact discrete references, independent of the neural field.
ema <- vmar_memory_create("ema", dimension=2, decay=0.5)
ema <- vmar_memory_update(ema, c(2,4))
ema <- vmar_memory_update(ema, c(4,8))
stopifnot(isTRUE(all.equal(ema$value,c(2.5,5))))
relax <- vmar_memory_relaxation(c(0,0), c(4,8), 0.2, 0.1)
stopifnot(isTRUE(all.equal(relax,c(4,8)*(1-exp(-2)))))
cat("G0.5-E.3 discrete memory references: PASS\n")
