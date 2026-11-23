# G0.5-E.5.4.3: Adam is an external discrete-time reference, NOT a core policy.
# Execute from the C project root after loading vMAR.Core >= 0.0.12.
if (!exists("vmar_custom_neuron_loss_gradient",mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
make_spec <- function(model) {
  if (model=="tanh") return(vmar_activation_spec("tanh",
      function(z,p) tanh(z),function(z,p) 1-tanh(z)^2))
  if (model=="prelu") return(vmar_activation_spec("prelu",
      function(z,p) ifelse(z>0,z,p[1]*z),
      function(z,p) ifelse(z>0,1,p[1]),parameters=0.1,
      derivative_parameters=function(z,p) matrix(ifelse(z<0,z,0),ncol=1),
      trainable=TRUE))
  vmar_activation_spec("swish",function(z,p) z*stats::plogis(p[1]*z),
      function(z,p) {s<-stats::plogis(p[1]*z);s+p[1]*z*s*(1-s)},
      parameters=0.8,derivative_parameters=function(z,p) {
        s<-stats::plogis(p[1]*z);matrix(z^2*s*(1-s),ncol=1)
      },trainable=TRUE)
}

# Atomic Adam recurrence, full-batch, no schedule, no decay and no clipping.
# This is the original Adam update, not AdamW or AMSGrad.
adam_reference <- function(theta,gradient,m,v,k,lr,beta1,beta2,eps) {
  stopifnot(k>=1L,lr>0,beta1>=0,beta1<1,beta2>=0,beta2<1,
            eps>0,length(theta)==length(gradient),
            length(theta)==length(m),length(theta)==length(v),
            all(is.finite(c(theta,gradient,m,v))))
  mnew <- beta1*m+(1-beta1)*gradient
  vnew <- beta2*v+(1-beta2)*gradient^2
  mh <- mnew/(-expm1(k*log(beta1)))
  vh <- vnew/(-expm1(k*log(beta2)))
  updated <- theta-lr*mh/(sqrt(vh)+eps)
  stopifnot(all(is.finite(c(updated,mnew,vnew))))
  list(theta=updated,m=mnew,v=vnew)
}

# Mathematically exact EMA correspondence only for constant gradient and
# initial memory zero. It does NOT imply optimizer or trajectory equivalence.
h <- 0.1; tau <- -h/log(0.9); beta <- exp(-h/tau)
m <- 0
for (k in seq_len(20L)) m <- beta*m+(1-beta)*2
stopifnot(abs(m-2*(1-exp(-20*h/tau)))<1e-12)

run_case <- function(model,perturb,lr,beta1) {
  spec <- make_spec(model)
  theta <- c(0.2+perturb,-0.4,0.15,
             if(spec$trainable)spec$parameters else numeric())
  d <- length(theta);m <- v <- numeric(d)
  initial_loss <- vmar_custom_neuron_loss_gradient(x,target,theta,spec)$loss
  for(k in seq_len(40L)) {
    g <- vmar_custom_neuron_loss_gradient(x,target,theta,spec)$gradient
    step <- adam_reference(theta,g,m,v,k,lr,beta1,0.999,1e-8)
    theta <- step$theta;m <- step$m;v <- step$v
  }
  final_loss <- vmar_custom_neuron_loss_gradient(x,target,theta,spec)$loss
  values <- c(theta,rep(0,4L-d))
  data.frame(model=model,perturb=perturb,lr=lr,beta1=beta1,
    beta2=0.999,epsilon=1e-8,dimension=d,
    gradient_evaluations=40L,updates=40L,initial_loss=initial_loss,
    final_loss=final_loss,theta1=values[1],theta2=values[2],
    theta3=values[3],theta4=values[4],
    m1=m[1],m2=m[2],m3=m[3],m4=if(d==4)m[4] else 0,
    v1=v[1],v2=v[2],v3=v[3],v4=if(d==4)v[4] else 0)
}
rows <- list()
for(model in c("tanh","prelu","swish"))
  for(perturb in c(-0.1,0.1))
    for(lr in c(0.001,0.01)) for(beta1 in c(0.8,0.9))
      rows[[length(rows)+1L]] <- run_case(model,perturb,lr,beta1)
study <- do.call(rbind,rows)
stopifnot(nrow(study)==24L,all(study$gradient_evaluations==40L),
          all(study$updates==40L),all(is.finite(study$final_loss)))
dir.create("reference/e543",showWarnings=FALSE,recursive=TRUE)
utils::write.csv(study,"reference/e543/adam_sensitivity.csv",row.names=FALSE)
utils::write.csv(study,"reference/vmar_adam_parity.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.5.4.3 Adam reference:",nrow(study),
    "discrete runs, 40 full-batch gradient evaluations each\n")
print(aggregate(cbind(initial_loss,final_loss)~model+lr+beta1,
                study,mean),digits=7)
cat("Sensitivity grid, not hyperparameter optimization.\n")
if (file.exists("reference/e542/transfer_results.csv")) {
  old <- utils::read.csv("reference/e542/transfer_results.csv")
  stopifnot(all(old$gradient_evaluations==40L))
  cat("Previous vMAR trajectories also used 40 gradient evaluations,",
      "but their physical time, initialization of memory,",
      "and update counts are not Adam-equivalent.\n")
}
cat("Claim gate: no assertion of universal superiority or exact Adam equivalence.\n")
