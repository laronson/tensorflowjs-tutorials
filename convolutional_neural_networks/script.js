import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js";
import { normalize } from "./normalize.js";
import { train } from "./train.js";
import { evaluate } from "./evaluate.js";
import { drawImage } from "./drawImage.js";
import { createModel } from "./createModel.js";

// Grab a reference to the MNIST input values (pixel data).
const INPUTS = TRAINING_DATA.inputs;

// Grab reference to the MNIST output values.
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays to remove any order, but do so in the same way so
// inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array is 2 dimensional.
const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);

// Output feature Array is 1 dimensional.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

const model = createModel();

await train(model, INPUTS_TENSOR, OUTPUTS_TENSOR, INPUTS);
OUTPUTS_TENSOR.dispose();
INPUTS_TENSOR.dispose();

var interval = 2000;
// Perform a new classification after a certain interval.
setInterval(async () => {
  const { OFFSET } = await evaluate(model, INPUTS, OUTPUTS);
  drawImage(INPUTS[OFFSET]);
}, interval);

const RANGER = document.getElementById("ranger");
const DOM_SPEED = document.getElementById("domSpeed");

// When user drags slider update interval.
RANGER.addEventListener("input", function (e) {
  interval = this.value;
  DOM_SPEED.innerText =
    "Change speed of classification! Currently: " + interval + "ms";
});
