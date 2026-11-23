# G0.5-E.5.4.4 — observable second moment and bounded feedback ablation.
# Run from C vMAR project root, with vMAR.Core >= 0.0.12 loaded.
if(!exists("vmar_custom_neuron_loss_gradient",mode="function"))library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
specification <- function(model) {
  if(model=="tanh")return(vmar_activation_spec("tanh",function(z,p)tanh(z),
    function(z,p)1-tanh(z)^2))
  if(model=="prelu")return(vmar_activation_spec("prelu",
    function(z,p)ifelse(z>0,z,p[1]*z),
    function(z,p)ifelse(z>0,1,p[1]),parameters=0.1,
    derivative_parameters=function(z,p)matrix(ifelse(z<0,z,0),ncol=1),
    trainable=TRUE))
  vmar_activation_spec("swish",function(z,p)z*stats::plogis(p[1]*z),
    function(z,p){s<-stats::plogis(p[1]*z);s+p[1]*z*s*(1-s)},
    parameters=0.8,derivative_parameters=function(z,p){
      s<-stats::plogis(p[1]*z);matrix(z^2*s*(1-s),ncol=1)
    },trainable=TRUE)
}
second_field <- function(spec,mode) {
  d <- 3L+if(spec$trainable)length(spec$parameters)else 0L
  function(t,state) {
    stopifnot(length(state)==(if(mode=="m1")2L else 3L)*d,
              all(is.finite(state)))
    theta<-state[seq_len(d)];m<-state[d+seq_len(d)]
    g<-vmar_custom_neuron_loss_gradient(x,target,theta,spec)$gradient
    if(mode=="m1")return(c(-0.3*(g+0.2*m),(g-m)/0.8))
    v<-state[2L*d+seq_len(d)]
    stopifnot(all(v>=0))
    coupling<-if(mode=="m2c")0.2/(1+sqrt(mean(v))/0.1)else 0.2
    c(-0.3*(g+coupling*m),(g-m)/0.8,(g*g-v)/0.4)
  }
}
run_case <- function(model,method,perturb,mode) {
  spec<-specification(model)
  theta<-c(0.2+perturb,-0.4,0.15,
           if(spec$trainable)spec$parameters else numeric())
  d<-length(theta)
  state<-c(theta,rep(0,d),if(mode!="m1")rep(0,d)else numeric())
  initial_loss<-vmar_custom_neuron_loss_gradient(x,target,theta,spec)$loss
  n<-if(method=="euler")40L else 10L
  advance<-if(method=="euler")vmar_euler_step else vmar_rk4_step
  field<-second_field(spec,mode)
  # Every field callback evaluates the analytical gradient exactly once.
  for(k in seq_len(n))state<-advance(field,(k-1L)/n,state,1/n)
  final_loss<-vmar_custom_neuron_loss_gradient(
    x,target,state[seq_len(d)],spec)$loss
  stopifnot(all(is.finite(state)),is.finite(final_loss),
            all(state[2L*d+seq_len(d)]>=0) || mode=="m1")
  theta4<-c(state[seq_len(d)],rep(0,4L-d))
  m4<-c(state[d+seq_len(d)],rep(0,4L-d))
  v4<-if(mode=="m1")rep(0,4L) else
    c(state[2L*d+seq_len(d)],rep(0,4L-d))
  data.frame(model=model,method=method,perturb=perturb,mode=mode,
    dimension=d,final_time=1,gradient_evaluations=40L,steps=n,
    initial_loss=initial_loss,final_loss=final_loss,
    v_mean=mean(v4[seq_len(d)]),
    theta1=theta4[1],theta2=theta4[2],theta3=theta4[3],theta4=theta4[4],
    m1=m4[1],m2=m4[2],m3=m4[3],m4=m4[4],
    v1=v4[1],v2=v4[2],v3=v4[3],v4=v4[4])
}
cases<-list()
for(model in c("tanh","prelu","swish"))
  for(method in c("euler","rk4"))for(perturb in c(-0.1,0.1))
    for(mode in c("m1","m2","m2c"))
      cases[[length(cases)+1L]]<-run_case(model,method,perturb,mode)
study<-do.call(rbind,cases)
stopifnot(nrow(study)==36L,all(study$gradient_evaluations==40L),
          all(study$final_time==1))
wide<-reshape(study[,c("model","method","perturb","mode","final_loss")],
    idvar=c("model","method","perturb"),timevar="mode",direction="wide")
wide$control_minus_observe<-wide$final_loss.m2c-wide$final_loss.m2
wide$observe_minus_m1<-wide$final_loss.m2-wide$final_loss.m1
stopifnot(nrow(wide)==12L,max(abs(wide$observe_minus_m1))<1e-12)
dir.create("reference/e544",recursive=TRUE,showWarnings=FALSE)
utils::write.csv(study,"reference/e544/second_moment_results.csv",row.names=FALSE)
utils::write.csv(wide,"reference/e544/paired_contrasts.csv",row.names=FALSE)
parity<-subset(study,mode=="m2c")
utils::write.csv(parity,"reference/vmar_second_moment_parity.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.5.4.4:",nrow(study),"fixed-time/equal-gradient cases;",
    nrow(wide),"paired contrasts;",nrow(parity),"R-C23 parity cases\n")
print(aggregate(cbind(final_loss,v_mean)~model+method+mode,
                study,mean),digits=8)
cat("M1 = M2 observational theta/m trajectory: PASS\n")
cat("M2C vs M2 better:",sum(wide$control_minus_observe< -1e-12),
    "ties:",sum(abs(wide$control_minus_observe)<=1e-12),
    "worse:",sum(wide$control_minus_observe>1e-12),"\n")
cat("Interpretation: a single small dataset cannot establish general utility.\n")
