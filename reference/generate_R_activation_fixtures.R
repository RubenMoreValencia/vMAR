# G0.5-E.5: run from the C project root after loading vMAR.Core >= 0.0.9.
if (!exists("vmar_activation_eval", mode="function")) {
  if (requireNamespace("vMAR.Core",quietly=TRUE)) library(vMAR.Core)
}
if (!exists("vmar_activation_eval",mode="function"))
  stop("Load the updated vMAR.Core 0.0.9 with devtools::load_all() first")

x <- matrix(c(1,0.5,-0.5,1.5,0.7,-1),nrow=3,byrow=TRUE)
target <- c(0.25,-0.3,0.7)
theta0 <- c(0.2,-0.4,0.15)
rows <- list()
for (activation in c("tanh","sigmoid","identity","softplus","relu",
                     "leaky_relu")) {
  for (method in c("euler","rk4")) {
    for (dt in c(0.1,0.025)) {
      for (n_steps in c(1L,12L)) {
        initial <- vmar_neuron_loss_gradient(x,target,theta0,activation)
        field <- vmar_neuron_field(x,target,0.3,activation)
        advance <- if(method=="euler") vmar_euler_step else vmar_rk4_step
        theta <- theta0
        for (k in seq_len(n_steps))
          theta <- advance(field,(k-1L)*dt,theta,dt)
        final <- vmar_neuron_loss_gradient(x,target,theta,activation)
        rows[[length(rows)+1L]] <- data.frame(
          id=paste(activation,method,sprintf("h%.3f",dt),
                   paste0("n",n_steps),sep="_"),
          activation=activation,integrator=method,eta=0.3,
          step=dt,n_steps=n_steps,
          loss_initial=initial$loss,
          grad1=initial$gradient[1],grad2=initial$gradient[2],
          grad3=initial$gradient[3],
          prediction1=initial$prediction[1],
          prediction2=initial$prediction[2],
          prediction3=initial$prediction[3],
          theta1=theta[1],theta2=theta[2],theta3=theta[3],
          loss_final=final$loss)
      }
    }
  }
}

fixtures <- do.call(rbind,rows)
stopifnot(nrow(fixtures)==48L)
utils::write.csv(fixtures,"reference/vmar_activation_parity.csv",
                 row.names=FALSE,quote=FALSE)
cat("G0.5-E.5 activation fixtures generated:",nrow(fixtures),"cases\n")
print(aggregate(loss_final~activation+integrator,data=fixtures,FUN=mean),
      row.names=FALSE,digits=12)
