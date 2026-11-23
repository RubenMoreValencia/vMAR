# G0.5-E.5.4.2 — bounded transfer and robustness audit.
# Run from C vMAR project root with vMAR.Core >= 0.0.12 loaded.
if (!exists("vmar_regime_diagnose",mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
make_spec <- function(model) {
  if(model=="tanh") return(vmar_activation_spec("tanh",
    function(z,p) tanh(z),function(z,p) 1-tanh(z)^2))
  if(model=="prelu") return(vmar_activation_spec("prelu",
    function(z,p) ifelse(z>0,z,p[1]*z),
    function(z,p) ifelse(z>0,1,p[1]),parameters=0.1,
    derivative_parameters=function(z,p) matrix(ifelse(z<0,z,0),ncol=1),
    trainable=TRUE))
  vmar_activation_spec("swish",function(z,p) z*stats::plogis(p[1]*z),
    function(z,p) {s<-stats::plogis(p[1]*z);s+p[1]*z*s*(1-s)},
    parameters=0.8,
    derivative_parameters=function(z,p) {
      s<-stats::plogis(p[1]*z);matrix(z^2*s*(1-s),ncol=1)
    },trainable=TRUE)
}
run_case <- function(model,method,policy,perturb,memory0) {
  spec <- make_spec(model)
  theta <- c(0.2+perturb,-0.4,0.15,
             if(spec$trainable)spec$parameters else numeric())
  d <- length(theta)
  state <- c(theta,rep(memory0,d));initial_loss <-
    vmar_custom_neuron_loss_gradient(x,target,theta,spec)$loss
  n <- if(method=="euler") 20L else 8L
  h <- 1/n;couplings <- contributions <- numeric(n)
  positive <- numeric(n)
  advance <- if(method=="euler") vmar_euler_step else vmar_rk4_step
  for(k in seq_len(n)) {
    # Every policy pays one diagnostic gradient evaluation per step.
    dgn <- vmar_regime_diagnose(x,target,spec,state,0.2)
    c_used <- switch(policy,observe=0,fixed=0.2,adaptive=dgn$coupling)
    couplings[k] <- c_used
    positive[k] <- as.integer(c_used>1e-14)
    contributions[k] <- if(dgn$gradient_norm>1e-12)
      c_used*dgn$memory_norm/dgn$gradient_norm else 0
    field <- vmar_adaptive_field(x,target,spec,eta=0.3,tau=0.8,
                                 coupling=c_used)
    state <- advance(field,(k-1L)*h,state,h)
  }
  final_loss <- vmar_custom_neuron_loss_gradient(
    x,target,state[seq_len(d)],spec)$loss
  values <- c(state,rep(0,8-length(state)))
  data.frame(model=model,method=method,policy=policy,perturb=perturb,
    memory0=memory0,final_time=1,gradient_evaluations=40L,
    initial_loss=initial_loss,final_loss=final_loss,
    intervention_rate=mean(positive),mean_coupling=mean(couplings),
    relative_contribution=mean(contributions),dimension=d,
    state1=values[1],state2=values[2],state3=values[3],
    state4=values[4],state5=values[5],state6=values[6],
    state7=values[7],state8=values[8])
}
rows <- list()
for(model in c("tanh","prelu","swish"))
  for(method in c("euler","rk4"))
    for(perturb in c(-0.1,0.1)) for(memory0 in c(-0.8,0.8))
      for(policy in c("observe","fixed","adaptive"))
        rows[[length(rows)+1L]] <- run_case(model,method,policy,
                                             perturb,memory0)
study <- do.call(rbind,rows)
stopifnot(nrow(study)==72L,all(study$final_time==1),
          all(study$gradient_evaluations==40L),
          all(is.finite(study$final_loss)),
          all(study$intervention_rate>=0&study$intervention_rate<=1),
          all(study$mean_coupling>=0&study$mean_coupling<=0.2+1e-12))

# Paired differences are computed only within identical method/model/initial
# conditions, so no conclusion relies on unpaired means.
wide <- reshape(study[,c("model","method","perturb","memory0",
                          "policy","final_loss")],
  idvar=c("model","method","perturb","memory0"),
  timevar="policy",direction="wide")
wide$adaptive_minus_observe <- wide$final_loss.adaptive-wide$final_loss.observe
wide$adaptive_minus_fixed <- wide$final_loss.adaptive-wide$final_loss.fixed
stopifnot(nrow(wide)==24L)
dir.create("reference/e542",showWarnings=FALSE,recursive=TRUE)
utils::write.csv(study,"reference/e542/transfer_results.csv",row.names=FALSE)
utils::write.csv(wide,"reference/e542/paired_differences.csv",row.names=FALSE)
parity <- subset(study,policy=="adaptive")
utils::write.csv(parity[,c("model","method","perturb","memory0",
  "dimension","final_loss","intervention_rate","mean_coupling",
  "relative_contribution",paste0("state",1:8))],
  "reference/vmar_transfer_parity.csv",row.names=FALSE,quote=FALSE)
cat("G0.5-E.5.4.2:",nrow(study),"equal-time/equal-budget cases,",
    nrow(wide),"paired contrasts,",nrow(parity),"R-C23 cases\n")
print(aggregate(cbind(final_loss,intervention_rate,mean_coupling)~
  model+method+policy,data=study,FUN=mean),digits=6)
cat("Pairs adaptive < observe:",sum(wide$adaptive_minus_observe< -1e-12),
    "| ties:",sum(abs(wide$adaptive_minus_observe)<=1e-12),
    "| worse:",sum(wide$adaptive_minus_observe>1e-12),"\n")
cat("No universal superiority conclusion: one small dataset, 3 activations.\n")
