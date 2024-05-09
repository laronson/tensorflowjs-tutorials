export function defineClassificationModelHead(numberOfOutputOptions) {
  let classificationModel = tf.sequential();

  //Add hidden layer for first layer of multi-layer perceptron classification model
  classificationModel.add(
    tf.layers.dense({
      inputShape: [1024], //since this is the head of the classification model we want a flattened input shape
      units: 128,
      activation: "relu",
    })
  );

  classificationModel.add(
    tf.layers.dense({ units: numberOfOutputOptions, activation: "softmax" })
  );

  classificationModel.summary();

  classificationModel.compile({
    optimizer: "adam", //Automatically changes our learning rate for us over time
    loss:
      numberOfOutputOptions === 2
        ? "binaryCrossentropy"
        : "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return classificationModel;
}
