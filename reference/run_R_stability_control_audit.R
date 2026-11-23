# G0.5-E.4.1: numerical audit, executed from the C vMAR project directory.
# The reference uses only documented vMAR.Core neuron/integrator functions.
if (!exists("vmar_neuron_loss_gradient", mode="function")) library(vMAR.Core)

x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
rows <- list()

for (activation in c("tanh","sigmoid")) {
  grad_loss <- function(state) vmar_neuron_loss_gradient(
    x,target,state[1:3],activation)
  for (method in c("euler","rk4")) {
    advance <- if (method=="euler") vmar_euler_step else vmar_rk4_step
    for (scenario in c("baseline","observe","coupled","control")) {
      for (perturb in c(-0.1,0,0.1)) {
        is_control <- scenario=="control"
        coupling <- if (scenario %in% c("coupled","control")) 0.2 else 0
        state <- c(0.2+perturb,-0.4,0.15,
                   if (scenario=="baseline") 0 else 0.8,0,0)
        initial <- state
        field <- function(t,s) {
          g <- grad_loss(s)$gradient
          c(-0.3*(g+coupling*s[4:6]),(g-s[4:6])/0.6)
        }
        final_time <- 0
        accepted <- 0L; rejected <- 0L; evaluations <- 0L
        max_theta <- max(abs(state[1:3]))
        max_memory <- max(abs(state[4:6]))
        if (!is_control) {
          for (k in seq_len(20L)) {
            state <- advance(field,(k-1)*0.05,state,0.05)
            evaluations <- evaluations+if (method=="euler") 1L else 4L
            final_time <- k*0.05
            accepted <- accepted+1L
            max_theta <- max(max_theta,abs(state[1:3]))
            max_memory <- max(max_memory,abs(state[4:6]))
          }
        } else {
          for (k in seq_len(20L)) {
            old <- grad_loss(state)$loss
            h <- 0.2
            if (max(abs(state[4:6]))>0.5) h <- h*0.5
            for (attempt in seq_len(6L)) {
              candidate <- advance(field,final_time,state,h)
              new <- grad_loss(candidate)$loss
              evaluations <- evaluations+if (method=="euler") 1L else 4L
              if (new<=old) {
                state <- candidate
                final_time <- final_time+h
                accepted <- accepted+1L
                max_theta <- max(max_theta,abs(state[1:3]))
                max_memory <- max(max_memory,abs(state[4:6]))
                break
              }
              rejected <- rejected+1L
              h <- h*0.5
            }
          }
        }
        loss <- grad_loss(state)$loss
        # Fine RK4 is a reference approximation, NOT an analytic exact solution.
        reference_error <- -1
        if (!is_control) {
          ref <- initial
          for (k in seq_len(200L))
            ref <- vmar_rk4_step(field,(k-1)*0.005,ref,0.005)
          reference_error <- sqrt(sum((state-ref)^2))
        }
        rows[[length(rows)+1L]] <- data.frame(
          id=paste(activation,method,scenario,
                   paste0("p",sprintf("%+.1f",perturb)),sep="_"),
          phase=if (is_control) "control" else "fixed_time",
          activation=activation,method=method,scenario=scenario,
          perturb=perturb,final_time=final_time,final_loss=loss,
          max_theta=max_theta,max_memory=max_memory,
          reference_error=reference_error,
          accepted=accepted,rejected=rejected,field_evaluations=evaluations,
          theta1=state[1],theta2=state[2],theta3=state[3],
          memory1=state[4],memory2=state[5],memory3=state[6])
      }
    }
  }
}

audit <- do.call(rbind,rows)
# Equal-field-evaluation budget: Euler 20 x 0.05 versus RK4 5 x 0.20,
# each reaching the same mathematical time T=1 at 20 field evaluations.
budget_rows <- list()
for (activation in c("tanh","sigmoid")) {
  for (scenario in c("baseline","observe","coupled")) {
    for (perturb in c(-0.1,0,0.1)) {
      coupling <- if (scenario=="coupled") 0.2 else 0
      initial <- c(0.2+perturb,-0.4,0.15,
                   if (scenario=="baseline") 0 else 0.8,0,0)
      field <- function(t,s) {
        g <- vmar_neuron_loss_gradient(x,target,s[1:3],activation)$gradient
        c(-0.3*(g+coupling*s[4:6]),(g-s[4:6])/0.6)
      }
      ref <- initial
      for (k in seq_len(200L))
        ref <- vmar_rk4_step(field,(k-1)*0.005,ref,0.005)
      for (method in c("euler","rk4")) {
        h <- if (method=="euler") 0.05 else 0.2
        n_steps <- if (method=="euler") 20L else 5L
        advance <- if (method=="euler") vmar_euler_step else vmar_rk4_step
        state <- initial
        max_theta <- max(abs(state[1:3])); max_memory <- max(abs(state[4:6]))
        for (k in seq_len(n_steps)) {
          state <- advance(field,(k-1)*h,state,h)
          max_theta <- max(max_theta,abs(state[1:3]))
          max_memory <- max(max_memory,abs(state[4:6]))
        }
        budget_rows[[length(budget_rows)+1L]] <- data.frame(
          id=paste(activation,method,scenario,
                   paste0("p",sprintf("%+.1f",perturb)),"budget",sep="_"),
          phase="equal_budget",activation=activation,method=method,
          scenario=scenario,perturb=perturb,final_time=1,
          final_loss=vmar_neuron_loss_gradient(x,target,state[1:3],activation)$loss,
          max_theta=max_theta,max_memory=max_memory,
          reference_error=sqrt(sum((state-ref)^2)),
          accepted=n_steps,rejected=0L,field_evaluations=20L,
          theta1=state[1],theta2=state[2],theta3=state[3],
          memory1=state[4],memory2=state[5],memory3=state[6])
      }
    }
  }
}
audit <- rbind(audit,do.call(rbind,budget_rows))
utils::write.csv(audit,"reference/vmar_stability_control_audit.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.4.1 audit completed:",nrow(audit),"cases\n")
cat("Fixed-time cases:",sum(audit$phase=="fixed_time"),
    "| controlled cases:",sum(audit$phase=="control"),
    "| equal-budget cases:",sum(audit$phase=="equal_budget"),"\n")
cat("Fixed-time numerical reference error by method:\n")
print(aggregate(reference_error~method,
                data=subset(audit,phase=="fixed_time"),
                FUN=function(v)c(mean=mean(v),max=max(v))))
cat("Controlled trajectory results by scenario and method:\n")
print(aggregate(cbind(final_loss,accepted,rejected,field_evaluations)~
                scenario+method,data=subset(audit,phase=="control"),FUN=mean))
cat("Equal-field-budget reference error by integrator:\n")
print(aggregate(reference_error~method,
                data=subset(audit,phase=="equal_budget"),FUN=mean))
