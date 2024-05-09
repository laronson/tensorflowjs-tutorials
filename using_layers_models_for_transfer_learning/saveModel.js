export async function saveModel(
  mobileNetModel,
  classificationHead,
  numberOfOutputOptions
) {
  //Create a new model that is a combination of the custom mobileNet Model we created our classification head
  const combinedModel = tf.sequential();
  combinedModel.add(mobileNetModel);
  combinedModel.add(classificationHead);

  combinedModel.compile({
    optimizer: "adam",
    loss:
      numberOfOutputOptions === 2
        ? "binaryCrossentropy"
        : "categoricalCrossentropy",
  });

  combinedModel.summary();

  //Download our model so we can use it later. This model contains all of our training data so we can import it anywhere
  //and have it work in the same way
  await combinedModel.save("downloads://my-model");
}
