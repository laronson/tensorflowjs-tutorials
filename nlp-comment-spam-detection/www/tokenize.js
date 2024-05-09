/**
 * Import the model's vocabulary file converted to JS so we can use it for encoding inputs to the model
 * This file essentially acts as a lookup table for words that the model knows and has internal embeddings for.  An
 * embedding is the list of values known to the model for each word that describes each word by the dimensions the model
 * uses to classify each word.  For example, a simple model that tries to understand words in terms of medical and
 * biological terms would have two dimensions, [medical, biological]. The word tumor may have an embedding of [.75, .75]
 * because it is both medical and biological by nature, however, the word x-ray may have an embedding of [.9, .01] because
 * it is mostly medical by nature.  These embeddings are stored in a special embeddings layer of the model but are accessed
 * via this vocabulary file at a higher level.
 */
import * as DICTIONARY from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/dictionary.js";

const ENCODING_LENGTH = 20;

/**
 * This function is responsible for creating the encoding for a given word array to be used as the input to the model.
 * The values contained in the encoding will be the numerical representation of each word in the word array as they
 * exist in the imported vocabulary file.  The returned encoding will be used by the model to look up the embeddings
 * for each word in the word array.
 */
export function tokenize(wordArray) {
  const returnArray = [DICTIONARY.START];

  wordArray.forEach((word) => {
    const encoding = DICTIONARY.LOOKUP[word];
    returnArray.push(encoding === undefined ? DICTIONARY.UNKNOWN : encoding);
  });

  //Most NLP models have a fixed length for how many words a string of text can be when it accepts a string for analysis
  //If the string is less than the specified length, the model expects any empty space to be filled with a predefined
  //PAD value stored in the vocabulary file.  This code performs such padding in the case where the return array is less
  //than the desired encoding length
  while (returnArray.length < ENCODING_LENGTH) {
    returnArray.push(DICTIONARY.PAD);
  }

  console.log([returnArray]);

  return tf.tensor2d([returnArray]);
}
