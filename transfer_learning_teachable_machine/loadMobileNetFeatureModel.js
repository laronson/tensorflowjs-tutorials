const STATUS = document.getElementById("status");

/**
 * Loads the mobile net feature model and warms it up so its ready for use.
 */
export async function loadMobileNetFeatureModel(inputHeight, inputWidth) {
  const URL =
    "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";

  const mobileNet = await tf.loadGraphModel(URL, { fromTFHub: true });
  STATUS.innerText = "MobileNet v3 loaded successfully!";

  //When importing and using larger models it can take some time to set everything up.  Therefore, we need to leave time
  //to "warm the model up".  To do this, we want to run a test sample through the model at the time of setup to allow for
  //the setup steps to occur now, before timing may become more critical when running actual data through the model.
  tf.tidy(function () {
    let answer = mobileNet.predict(tf.zeros([1, inputHeight, inputWidth, 3]));
    console.log(answer.shape);
  });

  return mobileNet;
}
