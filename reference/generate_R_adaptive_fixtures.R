# G0.5-E.5.2; run from C vMAR root after devtools::load_all() on R package.
if (!exists("vmar_adaptive_control_step",mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
spec <- vmar_activation_spec("prelu",function(z,p) ifelse(z>0,z,p[1]*z),
  function(z,p) ifelse(z>0,1,p[1]),parameters=0.1,
  derivative_parameters=function(z,p) matrix(ifelse(z<0,z,0),ncol=1),
  trainable=TRUE)
rows <- list()
for (method in c("euler","rk4")) for (coupling in c(0,0.2))
  for (memory0 in c(0,0.8)) for (step in c(0.1,4)) {
    state <- c(0.2,-0.4,0.15,0.1,rep(memory0,4))
    r <- vmar_adaptive_control_step(state,0,0,x,target,spec,
      integrator=method,step=step,reduction=0.5,max_attempts=4,
      loss_tolerance=0,memory_threshold=0.5,eta=0.3,tau=0.8,
      coupling=coupling)
    rows[[length(rows)+1L]] <- data.frame(
      id=paste(method,paste0("c",coupling),paste0("m",memory0),
        paste0("h",step),sep="_"),method=method,coupling=coupling,
      memory0=memory0,step=step,accepted=as.integer(r$accepted),
      attempts=r$attempts,rejected=r$rejected,
      field_evaluations=r$field_evaluations,effective_step=r$effective_step,
      final_time=r$time,loss=r$candidate_loss,
      theta1=r$state[1],theta2=r$state[2],theta3=r$state[3],
      theta4=r$state[4],memory1=r$state[5],memory2=r$state[6],
      memory3=r$state[7],memory4=r$state[8])
  }
fixture <- do.call(rbind,rows)
stopifnot(nrow(fixture)==16L)
utils::write.csv(fixture,"reference/vmar_adaptive_parity.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.5.2 adaptive R fixtures generated:",nrow(fixture),"cases\n")
print(fixture[,c("id","accepted","attempts","loss")],row.names=FALSE)
