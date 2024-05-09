import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";

//input feature pairs (house size & number of bedrooms)
const INPUTS = TRAINING_DATA.inputs; // number[][]

//current listed house prices in dollars given their features above (target output values you want to predict)
const OUTPUTS = TRAINING_DATA.outputs; // number[]

//shuffle the two arrays in the same way so inputs still match output indexes
tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

const { NORMALIZED_VALUES, MAX_VALUES, MIN_VALUES } = normalize(INPUTS_TENSOR);
postNormalization(NORMALIZED_VALUES, MAX_VALUES, MIN_VALUES, INPUTS_TENSOR);
const model = createModel();
await trainModel(model, NORMALIZED_VALUES, OUTPUTS_TENSOR);
predict(model, [[750, 1]], MIN_VALUES, MAX_VALUES);

//saveModelToLocalStorage(model)
cleanup(model, MIN_VALUES, MAX_VALUES);

//We must normalize our data so we are not consistantly dealing with extremelly large numbers.  To do this, we can
//convert each value in our tensors and normalize them to be between 0-1 with respect to each column of values contained
//in that tensor.  Min and max are passed as optional values to allow us to pass predicted min and max values when
//normalizing new data in the future so we do not need to explicitly find the min or max values in the future.
//normalize(tensor:Tensor, min?:number, max?:number): {NORMALIZED_VALUES:Tensor, MIN_VALUES:Tensor, MAX_VALUES:Tensor}
function normalize(tensor, min, max) {
  //tf.tidy will automatically run garbage collection on any extra tensors created in the function called inside of
  //tf.tidy and will only save the ones that are explicitly returned.  The function established within the call to tf.tidy
  //cannot be an async function
  const result = tf.tidy(function () {
    //The second param in these calls to tf.min and tf.max defines the axes.  If the rank of your tensor is greater than
    //one then you can specify what should be returned.  If you do not define a value of axes, the function will return
    //the smallest number it can find across the entire tensor regardless of its rank.  Here, we define the axes as 0 and
    //the function will therefore find the minimum value for all features within the tensor
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    //Subtract the MIN_VALUE from every value in the tensor and store the results in a new tensor.  This function will
    //subtract values with respect to their placement within the tensor of the tensor has a rank greater than 1.  For
    //example, [[1,1],[2,2]] - [1,1] = [[0,0],[1,1]]
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    //Calculate teh adjusted values devided by the range size as a new Tensor
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

function createModel() {
  //create the model architecture using tf.sequential to indicate that each neuron we specify for the model will
  //execute sequentially one after the other.
  const model = tf.sequential();

  //We will use one dense layer with 1 neuron (units parameter) and an input of 2 input feature values (representing
  //house size and number of rooms).  Notice that no activation function is specified here so it will use a "passthrough
  //activation function" (y=x) as the ouput of the neuron.  Also, each neuron will assign a single weight to each input
  //to the neuron meaning that the neuron is "densly connected" to those inputs.
  model.add(tf.layers.dense({ inputShape: [2], units: 1 }));

  model.summary();

  return model;
}

//Train the model.  You only need to train this model once
async function trainModel(model, NORMALIZED_VALUES, OUTPUTS_TENSOR) {
  //Choose a learning rate for training.  The proper learning rate can be determined through trial and error.  If you
  //set the learning rate too high, you may start to see NaN as prediction output values to indicate something went wrong.
  //If you see that you may want to consider decreasing your learning rate.
  const LEARNING_RATE = 0.01;

  //Compile the model while specifying the optimizer the model will use to hunt for the values for the wieghts and bias
  //values that the model will use.  We also specify the loss function the model will use to determine correctness.  In
  //this case, we specify the optimizer to be sgd which stands for stotastic gradient decent which is the mathimatical
  //algo used to determine the weights and bias keeping the desired learning rate in mind.  The loss function will use
  //the mean squared error to determine the loss function
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });

  //Train the model using the normalized values calculated from the inputs as well as the outputs.
  let results = await model.fit(NORMALIZED_VALUES, OUTPUTS_TENSOR, {
    validationSplit: 0.15, //Set asize 15% of the data for validation testing
    shuffle: true, //Make sure the data is shuffled in case it was still in order and not shuffled in pre-processing
    batchSize: 64, //Set a number for the number of values that can be calculated at one time so we do not calculate all at once
    epochs: 10, //Go through the data 10 times
  });

  OUTPUTS_TENSOR.dispose();
  NORMALIZED_VALUES.dispose();

  //you can get the average loss in the units of what is being calculated by square rooting the loss values stored in
  //model.history.loss and model.history.val_loss (for the validation stage).
  console.log(
    "Average error loss: ",
    Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );
  console.log(
    "Average validation error loss: ",
    Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  );
}

//Can call this as many times as you want after you have trained your model using the trainModel function above
function predict(model, new_inputs, MIN_VALUES, MAX_VALUES) {
  tf.tidy(function () {
    let { NORMALIZED_VALUES } = normalize(
      tf.tensor2d(new_inputs),
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

async function saveModelToLocalStorage(model) {
  //Save model files on local computer
  await model.save("downloads://my-model");

  //Save model to local storage for offline access
  await model.save("localstorage://demo/housePriceModel");
}

//If you are hosting the model somewhere on your own site, you can use a call to get the model using somthing similar
//to this
async function getSelfHostedModel() {
  const model = await tf.loadLayersModel("http://yoursite.com/model.json");
}

//If you have the model stored in local storage already, you can use a call to get the model using something similar
//to this
async function getModelFromLocalStorage() {
  const model = await tf.loadLayersModel("localstorage://demo/my-model");
}
