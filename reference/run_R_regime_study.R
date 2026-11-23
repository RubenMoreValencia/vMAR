# G0.5-E.5.4 reference experiment; run from vMAR C root with vMAR.Core
# 0.0.12 loaded. Comparison is descriptive, not evidence of superiority.
if(!exists("vmar_regime_control_step",mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
spec <- vmar_activation_spec("prelu",function(z,p) ifelse(z>0,z,p[1]*z),
  function(z,p) ifelse(z>0,1,p[1]),parameters=0.1,
  derivative_parameters=function(z,p) matrix(ifelse(z<0,z,0),ncol=1),
  trainable=TRUE)
loss <- function(s) vmar_custom_neuron_loss_gradient(x,target,s[1:4],spec)$loss
rows <- list(); study <- list()
for(method in c("euler","rk4")) for(cap in c(0,0.2))
  for(perturb in c(-0.1,0.1)) for(memory0 in c(-0.8,0.8))
    for(step in c(0.1,4)) {
      s0 <- c(0.2+perturb,-0.4,0.15,0.1,rep(memory0,4))
      r <- vmar_regime_control_step(s0,0,0,x,target,spec,
         max_coupling=cap,integrator=method,step=step,reduction=0.5,
         max_attempts=4L,memory_threshold=0.5,eta=0.3,tau=0.8)
      d <- r$diagnostic
      rows[[length(rows)+1L]] <- data.frame(
        method=method,max_coupling=cap,perturb=perturb,memory0=memory0,
        step=step,regime=d$regime,gradient_norm=d$gradient_norm,
        memory_norm=d$memory_norm,alignment=d$alignment,
        risk_ratio=d$risk_ratio,coupling=d$coupling,
        accepted=as.integer(r$accepted),attempts=r$attempts,
        rejected=r$rejected,field_evaluations=r$field_evaluations,
        final_time=r$time,candidate_loss=r$candidate_loss,
        theta1=r$state[1],theta2=r$state[2],theta3=r$state[3],
        theta4=r$state[4],memory1=r$state[5],memory2=r$state[6],
        memory3=r$state[7],memory4=r$state[8])

      # Compare three policies from the SAME confirmed state. Every adaptive
      # coupling is recalculated only after an accepted transition.
      for(policy in c("observe","fixed","adaptive")) {
        s <- s0; t <- 0; id <- 0L; rejected <- calls <- 0L
        for(k in seq_len(10L)) {
          step_result <- if(policy=="adaptive")
            vmar_regime_control_step(s,t,id,x,target,spec,
                max_coupling=cap,integrator=method,step=step,
                reduction=0.5,max_attempts=4L,memory_threshold=0.5,
                eta=0.3,tau=0.8) else
            vmar_adaptive_control_step(s,t,id,x,target,spec,
                integrator=method,step=step,reduction=0.5,
                max_attempts=4L,memory_threshold=0.5,eta=0.3,
                tau=0.8,coupling=if(policy=="observe") 0 else cap)
          rejected <- rejected+step_result$rejected
          calls <- calls+step_result$field_evaluations
          s <- step_result$state;t <- step_result$time;id <- step_result$step_id
          if(!step_result$accepted) break
        }
        study[[length(study)+1L]] <- data.frame(
          method=method,max_coupling=cap,perturb=perturb,
          memory0=memory0,step=step,policy=policy,accepted=id,
          rejected=rejected,field_evaluations=calls,final_time=t,
          loss=loss(s),initial_loss=loss(s0))
      }
    }
fixtures <- do.call(rbind,rows)
comparisons <- do.call(rbind,study)
stopifnot(nrow(fixtures)==32L,nrow(comparisons)==96L,
          all(fixtures$coupling>=0),
          all(fixtures$coupling<=fixtures$max_coupling+1e-14),
          all(fixtures$coupling[fixtures$alignment<=0]==0))
dir.create("reference/e54",recursive=TRUE,showWarnings=FALSE)
utils::write.csv(fixtures,"reference/vmar_regime_parity.csv",
                 row.names=FALSE,quote=FALSE)
utils::write.csv(comparisons,"reference/e54/policy_comparison.csv",
                 row.names=FALSE)
cat("G0.5-E.5.4:",nrow(fixtures),"diagnostic/parity cases,",
    nrow(comparisons),"policy trajectories\n")
print(aggregate(cbind(accepted,rejected,field_evaluations)~policy+method,
                data=comparisons,FUN=mean),digits=6)
cat("Warning: losses are not equal-time comparisons when final_time differs.\n")
