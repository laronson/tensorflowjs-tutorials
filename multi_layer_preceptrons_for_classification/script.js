//import MNIST image data consisting image of hand draw numbers.  The data has already been converted
//to numeric data nad has been pre-normalized
import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";
import { train } from "./train.js";
import { evaluate } from "./evaluate.js";
import { drawImage } from "./drawImage.js";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

//Make sure you shuffle your data before you select your validation data set.  If you want to shuffle after your select
//your validation dataset, it is possible that your validata dataset may come from a single spot in the unshuffled inputs
//and may skew your validation results as they may be biased
tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// For classification we want our outputs to be in the format of a 1-hot encoded array.  This means that the array will
//consist of the number of possible outputs and each index in the 1-hot encoded array will correspond to the value of the
//output.  In this example, we are trying to indicate of a picture displays the number 0-9.  A 1-hot encoded output for
//a positive result for the number 4 would look something like this: [0,0,0,0,1,0,0,0,0,0].
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

const model = tf.sequential();

// Our first layer will consist of 32 neurons and will accept 784 different inputs.  Each input corresponds to each pixel's
//color value within a MNIST 28x28px image.  The layer will use the relu activation function
model.add(
  tf.layers.dense({ inputShape: [784], units: 64, activation: "relu" })
);

//Add a hidden layer to add depth to the algo consisting of 16 neurons and that also uses the relu activation function
//Notes: originally started with 16 neurons in this layer but when increase to 32 the average loss per epoch got way better
//and the model finished training with a lower loss score in the end
model.add(tf.layers.dense({ units: 64, activation: "relu" }));

//Add the output layer that consists of 10 neurons, one for each possible output of the algorithm representing the numbers
//0-9.  This layer uses the softmax activation function to output a 1D tensor consisting of 10 values that all add up to
//1.  These values are indicitive of the confidence interval for the possible output of the function.  For example, if an
//an image is fed into the model that is most likely a 5 but may also be a 2, the output may look something like this
//[0,0,.2,0,0,.8,0,0,0,0,0].
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

const results = await train(model, INPUTS_TENSOR, OUTPUTS_TENSOR);

INPUTS_TENSOR.dispose();
OUTPUTS_TENSOR.dispose();

// const { OFFSET } = await evaluate(model, INPUTS, OUTPUTS);
// drawImage(INPUTS[OFFSET]);

setInterval(async () => {
  const { OFFSET } = await evaluate(model, INPUTS, OUTPUTS);
  drawImage(INPUTS[OFFSET]);
}, 3000);
