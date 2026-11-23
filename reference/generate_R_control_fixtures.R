# G0.5-E.4: run in RStudio, working directory at C vMAR root.
# Uses vMAR.Core for the neuron and integrators. The controller itself is
# defined here as an independent, explicit reference-policy specification.
if (!exists("vmar_neuron_loss_gradient", mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1), nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
cases <- list()
for (activation in c("tanh","sigmoid")) {
  for (method in c("euler","rk4")) {
    for (coupling in c(0,0.2)) {
      for (initial_step in c(0.1,10)) {
        eta <- 0.3; tau <- 0.6
        reduction <- 0.5; max_attempts <- 5L; threshold <- 0.5
        state <- c(0.2,-0.4,0.15,0.8,0,0)
        confirmed_time <- 0; step_id <- 0L
        loss_fn <- function(s) vmar_neuron_loss_gradient(
          x,target,s[1:3],activation)$loss
        field <- function(t,s) {
          grad <- vmar_neuron_loss_gradient(x,target,s[1:3],
                                            activation)$gradient
          c(-eta*(grad+coupling*s[4:6]),(grad-s[4:6])/tau)
        }
        old_loss <- loss_fn(state)
        step <- initial_step
        if (max(abs(state[4:6]))>threshold) step <- step*reduction
        advance <- if (method=="euler") vmar_euler_step else vmar_rk4_step
        accepted <- FALSE; attempts <- 0L; rejected <- 0L
        for (k in seq_len(max_attempts)) {
          candidate <- advance(field,confirmed_time,state,step)
          candidate_loss <- loss_fn(candidate)
          attempts <- k
          effective_step <- step
          if (candidate_loss <= old_loss) {
            state <- candidate
            confirmed_time <- confirmed_time + step
            step_id <- step_id+1L
            accepted <- TRUE
            break
          }
          rejected <- rejected+1L
          step <- step*reduction
        }
        cases[[length(cases)+1L]] <- data.frame(
          id=paste(activation,method,paste0("c",coupling),
                   paste0("h",initial_step),sep="_"),
          activation=activation,method=method,coupling=coupling,
          initial_step=initial_step,accepted=as.integer(accepted),
          attempts=attempts,rejected=rejected,
          old_loss=old_loss,candidate_loss=candidate_loss,
          effective_step=effective_step,final_time=confirmed_time,
          step_id=step_id,
          theta1=state[1],theta2=state[2],theta3=state[3],
          memory1=state[4],memory2=state[5],memory3=state[6])
      }
    }
  }
}
fixture <- do.call(rbind,cases)
utils::write.csv(fixture,"reference/vmar_control_parity.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.4 control reference: ",nrow(fixture)," cases\n",sep="")
print(fixture[,c("id","accepted","attempts","rejected",
                 "effective_step","candidate_loss")],row.names=FALSE)
