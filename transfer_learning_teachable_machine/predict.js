const VIDEO = document.getElementById("webcam");

export function predict(
  imageHeight,
  imageWidth,
  mobileNet,
  classificationModel
) {
  return tf.tidy(function () {
    //get frame and normalize
    const videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
    const resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [imageHeight, imageWidth],
      true //Align corners
    );

    const imageFeatures = mobileNet.predict(resizedTensorFrame.expandDims());
    const prediction = classificationModel.predict(imageFeatures).squeeze();

    //Remember, the output of the classification model is a one-hot array so the model's prediction is the index with
    //the highest value within the one-hot array.
    const highestIndex = prediction.argMax().arraySync();
    const oneHotPredictionArray = prediction.arraySync();

    return { highestIndex, confidence: oneHotPredictionArray[highestIndex] };
  });
}
