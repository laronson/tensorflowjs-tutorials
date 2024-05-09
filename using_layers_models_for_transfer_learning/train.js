/**
 *
 * @param {number[]} trainingInputs An array of arrays containing image data represented by an array containing 1024 numbers with values between 0-255
 * @param {number[]} trainingOutputs The training inputs correct class represented by a number which correlates to the class's index in the classNames array
 * @param {string[]} classNames The classes we are using as categories by which we are classifying our images
 */
export async function train(
  trainingInputs,
  trainingOutputs,
  classNames,
  classificationModel
) {
  tf.util.shuffleCombo(trainingInputs, trainingOutputs);

  const outputsAsTensor = tf.tensor1d(trainingOutputs, "int32");
  //Each value in the outputsAsTensor array is a single number.  Convert these numbers to a onehot representation of that
  //number. For example, if classNames.length == 5, the number 2 would be converted to [0,1,0,0,0]
  const oneHotOutputs = tf.oneHot(outputsAsTensor, classNames.length);

  //trainingInputs is passes as an array of tensors.  In order to use this training data with our model, we need to convert
  //it to a standard 2d tensor.  We can do this with the .stack() command which takes an array of tensors and stacks them
  //into a single usable tensor
  const inputsAsTensor = tf.stack(trainingInputs);

  //Fit the model to the training data
  const results = await classificationModel.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 5,
    callbacks: { onEpochEnd: logProgress },
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
}

function logProgress(epoch, logs) {
  console.log("Data for Epoch: " + epoch, logs);
}
