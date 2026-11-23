# G0.5-E.5.1: run from C vMAR root with R vMAR.Core >= 0.0.10 loaded.
if (!exists("vmar_activation_spec",mode="function")) library(vMAR.Core)
x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)

make_spec <- function(model,trainable) {
  if (model=="prelu")
    return(vmar_activation_spec("prelu",
      function(z,p) ifelse(z>0,z,p[1]*z),
      function(z,p) ifelse(z>0,1,p[1]),parameters=0.1,
      derivative_parameters=function(z,p) matrix(ifelse(z<0,z,0),ncol=1),
      trainable=trainable))
  if (model=="swish")
    return(vmar_activation_spec("swish",
      function(z,p) z*stats::plogis(p[1]*z),
      function(z,p) {s<-stats::plogis(p[1]*z);s+p[1]*z*s*(1-s)},
      parameters=0.8,
      derivative_parameters=function(z,p) {
        s<-stats::plogis(p[1]*z);matrix(z^2*s*(1-s),ncol=1)
      },trainable=trainable))
  vmar_activation_spec("sine",function(z,p) sin(z),
                       function(z,p) cos(z))
}

rows <- list()
for (variant in c("prelu_fixed","prelu_trainable", "swish_fixed",
                  "swish_trainable","sine")) {
  model <- sub("_(fixed|trainable)$","",variant)
  trainable <- grepl("_trainable$",variant)
  spec <- make_spec(model,trainable)
  theta0 <- c(0.2,-0.4,0.15,
              if(trainable) spec$parameters else numeric())
  for (method in c("euler","rk4")) {
    for (step in c(0.1,0.025)) {
      for (n_steps in c(1L,12L)) {
        initial <- vmar_custom_neuron_loss_gradient(x,target,theta0,spec)
        field <- vmar_custom_neuron_field(x,target,spec,eta=0.3)
        advance <- if (method=="euler") vmar_euler_step else vmar_rk4_step
        state <- theta0
        for (k in seq_len(n_steps))
          state <- advance(field,(k-1L)*step,state,step)
        final <- vmar_custom_neuron_loss_gradient(x,target,state,spec)
        rows[[length(rows)+1L]] <- data.frame(
          id=paste(variant,method,sprintf("h%.3f",step),
                   paste0("n",n_steps),sep="_"),
          model=model,trainable=as.integer(trainable),method=method,
          eta=0.3,step=step,n_steps=n_steps,
          parameter0=if(length(spec$parameters)) spec$parameters[1] else 0,
          loss_initial=initial$loss,
          grad1=initial$gradient[1],grad2=initial$gradient[2],
          grad3=initial$gradient[3],
          grad4=if(trainable) initial$gradient[4] else 0,
          pred1=initial$prediction[1],pred2=initial$prediction[2],
          pred3=initial$prediction[3],
          theta1=state[1],theta2=state[2],theta3=state[3],
          theta4=if(trainable) state[4] else 0,
          loss_final=final$loss)
      }
    }
  }
}
fixture <- do.call(rbind,rows)
stopifnot(nrow(fixture)==40L)
utils::write.csv(fixture,"reference/vmar_custom_activation_parity.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.5.1 custom activation fixtures generated:",nrow(fixture),"cases\n")
print(aggregate(loss_final~model+trainable+method,data=fixture,FUN=mean),
      row.names=FALSE,digits=12)
