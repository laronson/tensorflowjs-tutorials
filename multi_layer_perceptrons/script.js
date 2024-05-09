//In this particular example of calculating the values returned from the x^2 function, with the epoch value of the training
//model set to 200, we can see that the loss generated after evaluating each epoch settings at 30 around epoch 70 and does
//not get any better.  This means that we can try to increase the learning rate around epoch 70 which is done in the logProcess
//callback function.  Once we added more neurons, we wanted to decrease the learning rate even more because there are more
//changes that could be made across having more neurons
const LEARNING_RATE = 0.0000999999;
const OPTIMIZER = tf.train.sgd(LEARNING_RATE);

//In this example, inputs will be the numbers 1-20
const INPUTS = [];
for (let n = 1; n <= 20; n++) {
  INPUTS.push(n);
}

//Output will be the input value squared to create an exponential function
const OUTPUTS = [];
for (let n = 0; n < INPUTS.length; n++) {
  OUTPUTS.push(INPUTS[n] * INPUTS[n]);
}

const INPUTS_TENSOR = tf.tensor1d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);
INPUTS_TENSOR.print();
OUTPUTS_TENSOR.print();

const { NORMALIZED_VALUES, MAX_VALUES, MIN_VALUES } = normalize(INPUTS_TENSOR);
postNormalization(NORMALIZED_VALUES, MAX_VALUES, MIN_VALUES, INPUTS_TENSOR);
const model = createMultiLayerModel();
await trainModel(model, NORMALIZED_VALUES, OUTPUTS_TENSOR);
predict(model, [7], MIN_VALUES, MAX_VALUES);

//normalize(tensor:Tensor, min?:number, max?:number): {NORMALIZED_VALUES:Tensor, MIN_VALUES:Tensor, MAX_VALUES:Tensor}
function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });
  return result;
}

function postNormalization(
  NORMALIZED_VALUES,
  MAX_VALUES,
  MIN_VALUES,
  INPUTS_TENSOR
) {
  console.log("Normalized Values:");
  NORMALIZED_VALUES.print();
  console.log("Min Values:");
  MIN_VALUES.print();
  console.log("Max Values:");
  MAX_VALUES.print();

  //Now that we have normalized our input data we no longer need to store the inputs tensor in memory
  INPUTS_TENSOR.dispose();
}

function createMultiLayerModel() {
  const model = tf.sequential();

  //create the first layer of the model with three different neurons
  //Now, because we have three neurons in this layer, we are using the relu activation function to  ensure that a neuron
  //will only produce an output when it crosses a threshold making each time a neuron fires a more meaninful output
  model.add(
    tf.layers.dense({ inputShape: [1], units: 10, activation: "relu" })
  );

  model.add(tf.layers.dense({ units: 10, activation: "relu" }));

  //create the second layer of the model as the output layer
  model.add(tf.layers.dense({ units: 1 }));

  model.summary();

  return model;
}

//Train the model.  You only need to train this model once
async function trainModel(model, NORMALIZED_VALUES, OUTPUTS_TENSOR) {
  model.compile({
    optimizer: OPTIMIZER,
    loss: "meanSquaredError",
  });

  //Train the model using the normalized values calculated from the inputs as well as the outputs.
  let results = await model.fit(NORMALIZED_VALUES, OUTPUTS_TENSOR, {
    //validationSplit: 0.15, //Because our input dataset is very small, we want to put all of our input data towards training instead of validation
    shuffle: true, //Make sure the data is shuffled in case it was still in order and not shuffled in pre-processing
    batchSize: 2, //Because our input dataset is small, we want to set our batch size to be small as well
    epochs: 200, //Since our input dataset is small, we should iterate through our training data more times.  Will this cause overfitting though?
    callbacks: { onEpochEnd: logProgress }, //Specify a callback after we iterate through the dataset once
  });

  OUTPUTS_TENSOR.dispose();
  NORMALIZED_VALUES.dispose();

  console.log(
    "Average error loss: ",
    Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );
}

function logProgress(epoch, logs) {
  console.log("Data for epoch ", epoch, Math.sqrt(logs.loss));
  //We saw in the initial run of the algo that the loss function flattened at around 30 at the 70th epoch.  Because
  //of this, we decided to decrease the learning rate at that epoch to see if we could also decrease the loss due to
  //updating the algo weights more frequently
  // if (epoch == 70) {
  //   console.log("HERE");
  //   OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
  // }
}

//Can call this as many times as you want after you have trained your model using the trainModel function above
function predict(model, new_input, MIN_VALUES, MAX_VALUES) {
  tf.tidy(function () {
    let { NORMALIZED_VALUES } = normalize(
      tf.tensor1d(new_input),
      MIN_VALUES,
      MAX_VALUES
    );

    let output = model.predict(NORMALIZED_VALUES);
    output.print();
  });
}

function cleanup(model, MIN_VALUES, MAX_VALUES) {
  MIN_VALUES.dispose();
  MAX_VALUES.dispose();
  model.dispose();
  console.log("leftover Tensors:", tf.memory().numTensors);
}

//Overall it seems that this lesson was meant to teach about the ways of tweaking your model to add or subtract neurons
//and layers to decrease loss while training your model.  While tweaking these parameters, you need to keep in mind the
//learning rate the model is using because as more layers and neurons are added, more trainable parameters are added and
//because of this, you dont want to tweak the model too much.  The things you may want to consider when tweaking the
//optimization of your model are Layers used, neurons used, trainable parameters and loss.  You also may want to consider
//how the options you choose affect memory usage and time
