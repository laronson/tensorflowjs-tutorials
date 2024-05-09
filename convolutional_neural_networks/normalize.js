// Function to take a Tensor and normalize values
// with respect to each column of values contained in that Tensor.
export function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    const MIN_VALUES = tf.scalar(min);
    const MAX_VALUES = tf.scalar(max);

    // Now calculate subtract the MIN_VALUE from every value in the Tensor
    // And store the results in a new Tensor.
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // Calculate the range size of possible values.
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    // Calculate the adjusted values divided by the range size as a new Tensor.
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    // Return the important tensors.
    return NORMALIZED_VALUES;
  });
  return result;
}
