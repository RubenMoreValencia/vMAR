# G0.5-E.5.4.1: Effective use of adaptive memory, equal-time/equal-budget.
# Run from vMAR C root after loading vMAR.Core 0.0.12.
if (!exists("vmar_regime_diagnose",mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
spec <- vmar_activation_spec("prelu",function(z,p) ifelse(z>0,z,p[1]*z),
  function(z,p) ifelse(z>0,1,p[1]),parameters=0.1,
  derivative_parameters=function(z,p) matrix(ifelse(z<0,z,0),ncol=1),
  trainable=TRUE)
loss <- function(s) vmar_custom_neuron_loss_gradient(x,target,s[1:4],spec)$loss
initial <- function(perturb,m0) c(0.2+perturb,-0.4,0.15,0.1,rep(m0,4))

# Controlled trajectories deliberately keep their reached time: a rejected
# proposal stops the path and cannot be treated as t=1 performance.
run_control <- function(method,policy,perturb,m0,cap=0.2) {
  state <- initial(perturb,m0);time <- 0;id <- 0L
  rejected <- field_calls <- diagnostic_calls <- 0L
  trace <- list();status <- "max_steps"
  for(k in seq_len(160L)) {
    if(time>=1-1e-12){status <- "reached_target";break}
    d <- vmar_regime_diagnose(x,target,spec,state,cap)
    diagnostic_calls <- diagnostic_calls+1L
    c_used <- switch(policy,observe=0,fixed=cap,adaptive=d$coupling)
    h <- min(0.1,1-time)
    r <- vmar_adaptive_control_step(state,time,id,x,target,spec,
      integrator=method,step=h,reduction=0.5,max_attempts=4L,
      memory_threshold=0.5,eta=0.3,tau=0.8,coupling=c_used)
    contribution <- if(d$gradient_norm>1e-12)
      c_used*d$memory_norm/d$gradient_norm else 0
    trace[[length(trace)+1L]] <- data.frame(
      method=method,policy=policy,perturb=perturb,memory0=m0,
      proposal=k,time_before=time,step_proposed=h,
      c_used=c_used,alignment=d$alignment,regime=d$regime,
      intervention=as.integer(c_used>1e-14),
      relative_contribution=contribution,
      accepted=as.integer(r$accepted),rejected=r$rejected,
      effective_step=r$effective_step,time_after=r$time,
      loss_before=r$old_loss,loss_candidate=r$candidate_loss)
    rejected <- rejected+r$rejected;field_calls <- field_calls+r$field_evaluations
    state <- r$state;time <- r$time;id <- r$step_id
    if(!r$accepted){status <- "exhausted";break}
  }
  if(time>=1-1e-12)status <- "reached_target"
  ts <- do.call(rbind,trace)
  list(trace=ts,summary=data.frame(method=method,policy=policy,
    perturb=perturb,memory0=m0,status=status,reached_time=time,
    accepted=id,rejected=rejected,field_evaluations=field_calls,
    diagnostic_gradients=diagnostic_calls,
    intervention_rate=mean(ts$intervention),
    mean_coupling=mean(ts$c_used),
    mean_relative_contribution=mean(ts$relative_contribution),
    initial_loss=loss(initial(perturb,m0)),final_loss=loss(state)))
}

# Equal-horizon, equal-gradient-evaluation experiment. All three policies
# compute a diagnostic once per step, including the baselines, to equalize
# scientific observability cost. Euler: 20*(1+1)=40 gradient calculations;
# RK4: 8*(1+4)=40. Loss measurements are separately excluded.
run_budget <- function(method,policy,perturb,m0,cap=0.2) {
  n <- if(method=="euler") 20L else 8L
  h <- 1/n; state <- initial(perturb,m0); interventions <- numeric(n)
  contributions <- numeric(n)
  advance <- if(method=="euler") vmar_euler_step else vmar_rk4_step
  for(k in seq_len(n)) {
    d <- vmar_regime_diagnose(x,target,spec,state,cap)
    c_used <- switch(policy,observe=0,fixed=cap,adaptive=d$coupling)
    interventions[k] <- c_used
    contributions[k] <- if(d$gradient_norm>1e-12)
      c_used*d$memory_norm/d$gradient_norm else 0
    field <- vmar_adaptive_field(x,target,spec,eta=0.3,tau=0.8,
                                 coupling=c_used)
    state <- advance(field,(k-1L)*h,state,h)
  }
  data.frame(method=method,policy=policy,perturb=perturb,memory0=m0,
    final_time=1,field_evaluations=n*if(method=="euler") 1L else 4L,
    diagnostic_gradients=n,total_gradient_evaluations=40L,
    intervention_rate=mean(interventions>1e-14),
    mean_coupling=mean(interventions),
    mean_relative_contribution=mean(contributions),
    initial_loss=loss(initial(perturb,m0)),final_loss=loss(state),
    theta1=state[1],theta2=state[2],theta3=state[3],theta4=state[4],
    memory1=state[5],memory2=state[6],memory3=state[7],memory4=state[8])
}

controlled <- list();traces <- list();budget <- list()
for(method in c("euler","rk4")) for(perturb in c(-0.1,0.1))
  for(m0 in c(-0.8,0.8)) for(policy in c("observe","fixed","adaptive")) {
    a <- run_control(method,policy,perturb,m0)
    controlled[[length(controlled)+1L]] <- a$summary
    traces[[length(traces)+1L]] <- a$trace
    budget[[length(budget)+1L]] <- run_budget(method,policy,perturb,m0)
  }
controlled <- do.call(rbind,controlled)
traces <- do.call(rbind,traces)
budget <- do.call(rbind,budget)
stopifnot(nrow(controlled)==24L,nrow(budget)==24L,
          all(budget$final_time==1),
          all(budget$total_gradient_evaluations==40L),
          all(budget$intervention_rate>=0 & budget$intervention_rate<=1),
          all(budget$mean_coupling>=0 & budget$mean_coupling<=0.2+1e-12))
dir.create("reference/e541",showWarnings=FALSE,recursive=TRUE)
utils::write.csv(controlled,"reference/e541/controlled_summary.csv",row.names=FALSE)
utils::write.csv(traces,"reference/e541/intervention_trace.csv",row.names=FALSE)
utils::write.csv(budget,"reference/e541/equal_budget.csv",row.names=FALSE)
utils::write.csv(budget[,c("method","policy","perturb","memory0",
  "final_loss","intervention_rate","mean_coupling",
  "mean_relative_contribution","theta1","theta2","theta3","theta4",
  "memory1","memory2","memory3","memory4")],
  "reference/vmar_intervention_parity.csv",row.names=FALSE,quote=FALSE)
cat("G0.5-E.5.4.1:",nrow(controlled),"controlled runs,",
    nrow(traces),"diagnostic trace rows,",nrow(budget),
    "equal-budget / R-C23 parity cases\n")
print(aggregate(cbind(intervention_rate,mean_coupling,
  mean_relative_contribution)~policy+method,data=budget,FUN=mean),digits=6)
print(table(controlled$policy,controlled$status))
cat("Scientific gate: compare loss only at a shared reached time and budget.\n")
