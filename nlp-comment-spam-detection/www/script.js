import { handleCommentPost, handleRemoteComment } from "./handleComment.js";
import { loadModel } from "./loadModel.js";
import { predict } from "./predict.js";
import { tokenize } from "./tokenize.js";

const SPAM_THRESHOLD = 0.75;

const POST_COMMENT_BTN = document.getElementById("post");
const COMMENT_TEXT = document.getElementById("comment");
const PROCESSING_CLASS = "processing";

console.log(io);
var socket = io.connect();
socket.on("remoteComment", handleRemoteComment);

POST_COMMENT_BTN.addEventListener("click", onPostCommentClick);

const model = await loadModel();

async function onPostCommentClick() {
  const {
    lowercaseSentenceArray,
    listItemElement,
    currentComment,
    commentTime,
  } = handleCommentPost();

  if (lowercaseSentenceArray) {
    const resultsArray = await predict(model, tokenize(lowercaseSentenceArray));
    if (resultsArray[1] > SPAM_THRESHOLD) {
      listItemElement.classList.add("spam");
    } else {
      socket.emit("comment", {
        username: "userx",
        timestamp: commentTime,
        comment: currentComment,
      });
    }
    POST_COMMENT_BTN.classList.remove(PROCESSING_CLASS);
    COMMENT_TEXT.classList.remove(PROCESSING_CLASS);
  }
}
