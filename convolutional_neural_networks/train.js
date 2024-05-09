/**
 * After training the model with this configuration, you will most likely see that the models accuracy on predicting
 * items in the training data is higher than the accuracy achieved when predicting values from the validation dataset.
 * This may indicate that the model is overfitting to the training data.  To fix this, you could add an extra layer to
 * the model called dropout layers.  These dropout layers will deactivate a fraction of the output from the prior layer
 * during training by setting the selected values to zero.  This will help reduce the chance of the model overfitting
 * to the training data by making sure that the model focuses on the most relevant features while training while not
 * focusing on the values that show up infrequently and/or only in the training dataset.
 * The more widely used dropout values are .25 (25%) or .5 (50%) dropout
 */

export async function train(model, INPUTS_TENSOR, OUTPUTS_TENSOR, INPUTS) {
  // Compile the model with the defined optimizer and specify a loss function to use.
  model.compile({
    optimizer: "adam", // Adam changes the learning rate over time which is useful.
    loss: "categoricalCrossentropy", // As this is a classification problem, dont use MSE.
    metrics: ["accuracy"], // As this is a classifcation problem you can ask to record accuracy in the logs too!
  });

  //When received, our image inputs are in the form of a 2d tensor with each inner array containing 784 numbers to
  //represent each input image. Because our model expects inputs in the form of 28x28 images, we need to reshape our
  //input to match this format
  //The reshape parameters are as follows: //[numberOfInputs, desiredInputWidth, desiredInputHeight,channelCount]
  const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1]);

  // Finally do the training itself using the reshaped inputs
  let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
    shuffle: true, // Ensure data is shuffled again before using each time.
    validationSplit: 0.15,
    batchSize: 256,
    epochs: 30,
    callbacks: { onEpochEnd: logProgress },
  });

  RESHAPED_INPUTS.dispose();
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}
