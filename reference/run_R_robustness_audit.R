# G0.5-E.4.2 — controlled robustness study using the verified R core.
# Run with the working directory at the C vMAR project root.
if (!exists("vmar_neuron_loss_gradient",mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
rows <- vector("list",648L)
row_id <- 0L
for (activation in c("tanh","sigmoid")) {
  for (method in c("euler","rk4")) {
    advance <- if (method=="euler") vmar_euler_step else vmar_rk4_step
    for (perturb in c(-0.5,0,0.5)) {
      for (tau in c(0.1,0.6,2)) {
        for (coupling in c(0,0.2,0.8)) {
          for (initial_step in c(0.1,1,10)) {
            for (max_attempts in c(1L,6L)) {
              row_id <- row_id+1L
              state <- c(0.2+perturb,-0.4,0.15,0.8+perturb,0,0)
              time <- 0; step_id <- 0L
              loss <- function(s) vmar_neuron_loss_gradient(
                x,target,s[1:3],activation)$loss
              field <- function(t,s) {
                g <- vmar_neuron_loss_gradient(x,target,s[1:3],activation)$gradient
                c(-0.3*(g+coupling*s[4:6]),(g-s[4:6])/tau)
              }
              old <- loss(state)
              h <- initial_step
              if(max(abs(state[4:6]))>0.5) h <- h*0.5
              accepted <- FALSE; rejected <- 0L; attempts <- 0L
              candidate_loss <- old; effective_step <- NA_real_
              for (k in seq_len(max_attempts)) {
                candidate <- advance(field,time,state,h)
                candidate_loss <- loss(candidate)
                effective_step <- h
                attempts <- k
                if(candidate_loss <= old) {
                  accepted <- TRUE
                  state <- candidate
                  time <- time+h
                  step_id <- step_id+1L
                  break
                }
                rejected <- rejected+1L
                h <- h*0.5
              }
              final_loss <- loss(state)
              rows[[row_id]] <- data.frame(
                id=paste(activation,method,paste0("p",perturb),
                         paste0("tau",tau),paste0("c",coupling),
                         paste0("h",initial_step),paste0("n",max_attempts),sep="_"),
                activation=activation,method=method,
                perturb=perturb,tau=tau,coupling=coupling,
                initial_step=initial_step,max_attempts=max_attempts,
                accepted=as.integer(accepted),attempts=attempts,
                rejected=rejected,field_evaluations=attempts*
                  (if(method=="euler") 1L else 4L),
                effective_step=effective_step,old_loss=old,
                candidate_loss=candidate_loss,final_loss=final_loss,
                final_time=time,step_id=step_id,
                theta1=state[1],theta2=state[2],theta3=state[3],
                memory1=state[4],memory2=state[5],memory3=state[6])
            }
          }
        }
      }
    }
  }
}
results <- do.call(rbind,rows)
stopifnot(nrow(results)==648L)
recovered <- with(results,accepted==1L & rejected>0L)
exhausted <- with(results,accepted==0L)
stopifnot(any(recovered),any(exhausted),any(results$accepted==1L))
utils::write.csv(results,"reference/vmar_robustness_parity.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.4.2 robustness R study:",nrow(results),"cases\n")
cat("Accepted:",sum(results$accepted),"| exhausted:",sum(exhausted),
    "| recovered:",sum(recovered),
    "| rejected attempts:",sum(results$rejected),"\n")
cat("By method (descriptive only):\n")
print(aggregate(cbind(accepted,rejected,field_evaluations)~method,
                data=results,FUN=mean))
