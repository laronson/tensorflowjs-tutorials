/**
 * The model returned from this function (in its current state) will contain ~21000 trainable parameters which is a
 * pretty heavy workload, especially for the browser.  Most often, this type of model training would happen outside the
 * browser and most likely on a server (possibly leveraging tensorFlowJs and Node.js).  However, because this is just
 * practice, we are going to run this in the browser
 */

export function createModel() {
  // Now actually create and define model architecture.
  const model = tf.sequential();

  /**
   * Add the first convolutional layer to the neural network the following comments will describe the configuration:
   *
   * inputShape: Because this is the first layer in the network, we must define the input shape of the inputs to the
   * model
   * filters: the number of filters the layer will use in the convolution
   * kernelSize: this is the size of the filters that will be used to perform the image processing.  Use a single number
   * for a square (e.g. 3 for a 3x3 filter) and an array for a rectangle (e.g. [2,3] for a 2x3 filter)
   * strides: How much a filter should jump or slide across the image between filter comparisons.  If this value is set
   * to one, it means that the filter will evaluate every pixel, moving one pixel between every evaluation with each
   * filter if the filter is set to 2, then the filter will evaluate on every other pixel.
   * padding: Choose from a padding strategy to handle the case where a filter is at the edge of an image and needs to
   * fill empty space with pre-defined numbers.  The "same" option will fill empty space with 0 when encountered.
   * Alternatively, you could defined a specific number to use instead of 0.
   * activation: define an activation function
   */
  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1], //[width of image, height of image, number of color channels (1=greyscale, 3=rgb)]
      filters: 16,
      kernelSize: 3,
      strides: 1,
      padding: "same",
      activation: "relu", // Use relu here to ensure non-linear relations can be done on the training data
    })
  );

  //Add a max pooling layer to the network that will accept the outputs from the previously defined convolution layer.
  //This layer will take the inputs, and pick the "highest" values from each 2x2 frame within the input with a stride of two
  //resulting in a output half the size of the inputs (28x28 image -> 14x14 output).  This layer will in turn create
  //16 14x14 feature maps.
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  //Add a second convolution layer to evaluate the 14x14 outputs of the previous max pooling layer.  Because the inputs
  //to this layer are smaller than the previous convolution layer, we can use double the amount of filters in this layer
  //because there is less data to analyze
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      strides: 1,
      padding: "same",
      activation: "relu",
    })
  );

  //Add a second max pooling layer to shrink the size of the 32 inputs, selecting the "highest" values
  //from each 2x2 frame within the input.  This layer will in turn create 32 7x7 feature maps.
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  //Take all 32 7x7 feature maps from the output of the second max pooling layer and convert them to a single tensor to
  //be used as the input to the following multi-level perceptron portion of the network
  model.add(tf.layers.flatten());

  //Add a dense layer to the network to evaluate the input from the previous "flatten layer".  This layer will contain
  //128 neurons which will each accept all 128 input values passed to it by the previous layer
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));

  //finally add the output layer consisting of a 10 neuron layer utilizing the softmax activation function to output a
  //"likelyhook" array for each of the 10 different value options that could be predicted by the model.
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.summary();

  return model;
}
