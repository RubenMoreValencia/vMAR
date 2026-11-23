# G0.5-E.5.3: numerical audit; run from the C vMAR project root after
# devtools::load_all() on the R reference package vMAR.Core >= 0.0.11.2.
if (!exists("vmar_adaptive_field",mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
spec <- vmar_activation_spec("prelu",function(z,p) ifelse(z>0,z,p[1]*z),
  function(z,p) ifelse(z>0,1,p[1]),parameters=0.1,
  derivative_parameters=function(z,p) matrix(ifelse(z<0,z,0),ncol=1),
  trainable=TRUE)

initial <- function(perturb,memory0) c(0.2+perturb,-0.4,0.15,0.1,
                                     rep(memory0,4))
loss <- function(s) vmar_custom_neuron_loss_gradient(x,target,s[1:4],spec)$loss
evolve <- function(method,coupling,perturb,memory0,n_steps,h) {
  field <- vmar_adaptive_field(x,target,spec,eta=0.3,tau=0.8,
                               coupling=coupling)
  advance <- if(method=="euler") vmar_euler_step else vmar_rk4_step
  state <- initial(perturb,memory0)
  for(k in seq_len(n_steps)) state <- advance(field,(k-1L)*h,state,h)
  state
}
rows <- list(); budget <- list(); controlled <- list()
for(method in c("euler","rk4")) for(coupling in c(0,0.2))
  for(perturb in c(-0.1,0,0.1)) for(memory0 in c(0,0.8)) {
    # Reference is a high-resolution RK4 trajectory for the SAME field and
    # same initial condition. It is not an analytical exact solution.
    reference <- evolve("rk4",coupling,perturb,memory0,1000L,0.001)
    for(h in c(0.2,0.1)) {
      steps <- as.integer(round(1/h))
      s <- evolve(method,coupling,perturb,memory0,steps,h)
      rows[[length(rows)+1L]] <- data.frame(
        method=method,coupling=coupling,perturb=perturb,memory0=memory0,
        step=h,n_steps=steps,final_time=steps*h,
        field_evaluations=steps*if(method=="euler") 1L else 4L,
        loss=loss(s),reference_loss=loss(reference),
        theta_error=max(abs(s[1:4]-reference[1:4])),
        state_error=max(abs(s-reference)),
        theta1=s[1],theta2=s[2],theta3=s[3],theta4=s[4],
        memory1=s[5],memory2=s[6],memory3=s[7],memory4=s[8])
    }
    # Equal evaluations, equal final time, unequal internal step size.
    n_steps <- if(method=="euler") 40L else 10L
    s <- evolve(method,coupling,perturb,memory0,n_steps,1/n_steps)
    budget[[length(budget)+1L]] <- data.frame(
      method=method,coupling=coupling,perturb=perturb,memory0=memory0,
      horizon=1,field_evaluations=40L,loss=loss(s),
      theta_error=max(abs(s[1:4]-reference[1:4])),
      state_error=max(abs(s-reference)))
    # Controlled proposals: time and evaluation cost are recorded; this
    # is NOT an equal-time study when rejection prevents reaching the goal.
    state <- initial(perturb,memory0); time <- 0;step_id <- 0L
    attempts <- rejected <- field_calls <- 0L
    for(k in seq_len(10L)) {
      response <- vmar_adaptive_control_step(state,time,step_id,x,target,spec,
        integrator=method,step=0.1,reduction=0.5,max_attempts=4L,
        memory_threshold=0.5,eta=0.3,tau=0.8,coupling=coupling)
      attempts <- attempts+response$attempts
      rejected <- rejected+response$rejected
      field_calls <- field_calls+response$field_evaluations
      state <- response$state;time <- response$time;step_id <- response$step_id
      if(!response$accepted) break
    }
    controlled[[length(controlled)+1L]] <- data.frame(
      method=method,coupling=coupling,perturb=perturb,memory0=memory0,
      accepted=step_id,rejected=rejected,attempts=attempts,
      field_evaluations=field_calls,final_time=time,
      reached_target=abs(time-1)<1e-12,loss=loss(state),
      initial_loss=loss(initial(perturb,memory0)))
  }
fixed <- do.call(rbind,rows)
equal_budget <- do.call(rbind,budget)
control <- do.call(rbind,controlled)
stopifnot(nrow(fixed)==48L,nrow(equal_budget)==24L,nrow(control)==24L,
          all(fixed$final_time==1),
          all(equal_budget$field_evaluations==40L))
# Observational-memory invariant: when coupling=0, theta must not depend
# on the initial memory, even though the memory coordinates themselves do.
uncoupled <- subset(fixed,coupling==0)
for(method_i in c("euler","rk4")) for(perturb_i in c(-0.1,0,0.1))
  for(step_i in c(0.2,0.1)) {
    a <- subset(uncoupled,method==method_i & perturb==perturb_i &
                step==step_i & memory0==0)
    b <- subset(uncoupled,method==method_i & perturb==perturb_i &
                step==step_i & memory0==0.8)
    stopifnot(nrow(a)==1L,nrow(b)==1L,
              max(abs(unlist(a[1,paste0("theta",1:4)],use.names=FALSE)-
                      unlist(b[1,paste0("theta",1:4)],use.names=FALSE)))<1e-12)
  }
dir.create("reference/e53",showWarnings=FALSE,recursive=TRUE)
utils::write.csv(fixed,"reference/e53/fixed_time.csv",row.names=FALSE)
utils::write.csv(equal_budget,"reference/e53/equal_budget.csv",row.names=FALSE)
utils::write.csv(control,"reference/e53/controlled.csv",row.names=FALSE)
# C comparator consumes only deterministic end-state columns, independently
# computed on the C side. The numerical reference remains an R-only audit.
utils::write.csv(fixed[,c("method","coupling","perturb","memory0",
                      "step","n_steps","loss","theta1","theta2",
                      "theta3","theta4","memory1","memory2",
                      "memory3","memory4")],
                 "reference/vmar_coupling_parity.csv",row.names=FALSE,
                 quote=FALSE)
cat("G0.5-E.5.3 audit:",nrow(fixed),"fixed-time,",
    nrow(equal_budget),"equal-budget,",nrow(control),
    "controlled cases\n")
print(aggregate(cbind(theta_error,state_error)~method+coupling,
                data=fixed,FUN=mean),digits=6)
print(aggregate(cbind(accepted,rejected,field_evaluations)~method+coupling,
                data=control,FUN=mean),digits=6)
