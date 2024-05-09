import { customPrint } from "./customPrint.js";

const STATUS = document.getElementById("status");

/**
 * Loads the mobile net feature model and warms it up so its ready for use.
 */
export async function loadMobileNetFeatureModel(inputHeight, inputWidth) {
  const URL =
    "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";

  const mobileNet = await tf.loadLayersModel(URL);
  STATUS.innerText = "MobileNet v2 loaded successfully!";

  //Print the summary of the model on web page instead of in the console using the customPrint function
  mobileNet.summary(null, null, customPrint);

  //Because we are using a layer model instead of a graph model we are able to access the individual layers of the model
  //Here, we grab the last layer of the model by name
  const lastLayer = mobileNet.getLayer("global_average_pooling2d_1");

  //Looking at the original model, it has a final dense layer before the global_average_pooling2d_1 layer that is called
  //predictions.  With the following line of code, we establish a new model that does not include the final prediction
  //layer of the original mobilenet model because we are going to replace it with our own multi-layer preceptron
  //prediction layer that we are going to train ourselves.  If you look at the summary of the new model, the last layer
  //is now the global_average_pooling2d_1 layer instead of the prediction layer.
  const mobileNetBase = tf.model({
    inputs: mobileNet.inputs,
    outputs: lastLayer.output,
  });
  mobileNetBase.summary();

  //When importing and using larger models it can take some time to set everything up.  Therefore, we need to leave time
  //to "warm the model up".  To do this, we want to run a test sample through the model at the time of setup to allow for
  //the setup steps to occur now, before timing may become more critical when running actual data through the model.
  tf.tidy(function () {
    let answer = mobileNetBase.predict(
      tf.zeros([1, inputHeight, inputWidth, 3])
    );
    console.log(answer.shape);
  });

  return mobileNetBase;
}

//NOTE* if you want to set any layer of the base model to not be trainable at train time, you can set the layer to not
//be trainable by setting the flag layer.trainable = false.  This could be useful if you want to hold off on training
//a model until a later time within your program
//You can save a reference to the "later trainable" models in an array
//Eventually you can then revisit the references to the layers you set to not be trainable and flip their flags back
//to true and train again so those layers end up getting trained in the context of the newly trained rest of model
