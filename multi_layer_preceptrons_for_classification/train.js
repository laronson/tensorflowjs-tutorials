export async function train(model, INPUTS_TENSOR, OUTPUTS_TENSOR) {
  // Compile the model with adam optimizer and the categoricalCrossentropy loss function as well as defining a metrics
  //property to log while training the model.
  // The adam optimizer will change the learning rate over time to optimize the models learning.  It is set to work well
  //with image processing.  You could still use the sgd model here as well.
  // The categoricalCrossentropy is used at the loss function because here, the meanSquaredErrorLoss will no longer work.
  //because this is now a classification problem and not a linear regression problem.
  // The metrics property will allow us to log the accuracy of how many images the model predicted correctly
  //We can use this to measure the models validation accuracy which we want to see go up over time.
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true, //This config ensures that the data is shuffled every time we go through it for each epoch
    validationSplit: 0.2,
    batchSize: 512,
    epochs: 200,
    callbacks: { onEpochEnd: logProgress },
  });

  console.log(
    " validation error loss: ",
    results.history.val_loss[results.history.val_loss.length - 1]
  );
  console.log(
    "accuracy: ",
    results.history.acc[results.history.acc.length - 1]
  );

  return results;
}

function logProgress(epoch, log) {
  console.log({ epoch, loss: log.loss });
}
