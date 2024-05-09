const VIDEO = document.getElementById("webcam");

export function gatherData(mobileNet, imageHeight, imageWidth) {
  const imageFeatures = tf.tidy(function () {
    //Grab a frame of the webcam and return it as a tensor
    const videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
    console.log({ videoFrameAsTensor });

    //Resize the image to fit the expected image dimensions of the mobile net model.  Because our webcam image is
    //640x480, this resize will stretch our image so eventually we may want to crop our image to be a perfect square
    //instead of stretching
    const resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [imageHeight, imageWidth],
      true //Align corners
    );

    //Now we must normalize our tensor data.  Because we are using greyscale images for this model, we know that all
    //values in our tensor must be between 0-255.  Therefore, to normalize, we can divide each value in the tensor by
    //255
    const normalizedTensorFrame = resizedTensorFrame.div(255);

    return mobileNet.predict(normalizedTensorFrame.expandDims()).squeeze();
  });
  console.log({ imageFeatures });

  return { imageFeatures };
}
