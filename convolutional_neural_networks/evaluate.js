import { normalize } from "./normalize.js";

const PREDICTION_ELEMENT = document.getElementById("prediction");

// Map output index to label.
const LOOKUP = [
  "T-shirt",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];

export async function evaluate(model, INPUTS, OUTPUTS) {
  // Select a random index from all the example images we have in the training data arrays.
  const OFFSET = Math.floor(Math.random() * INPUTS.length);

  // Clean up created tensors automatically.
  let answer = tf.tidy(function () {
    let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255);

    //Like we reshaped the training data that was fed into our model to match the desired input format by our CNN to be
    //28x28 images, we also need to do this to any input we are trying to evaluate after the model has been trained
    const reshapedInput = newInput.reshape([1, 28, 28, 1]);

    let output = model.predict(reshapedInput);
    output.print();

    return output.squeeze().argMax();
  });

  const predictedIndex = await answer.array();

  PREDICTION_ELEMENT.innerText = LOOKUP[predictedIndex];
  PREDICTION_ELEMENT.setAttribute(
    "class",
    predictedIndex === OUTPUTS[OFFSET] ? "correct" : "wrong"
  );
  answer.dispose();

  return { OFFSET };
}
