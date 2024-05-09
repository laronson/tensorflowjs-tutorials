const POST_COMMENT_BTN = document.getElementById("post");
const COMMENT_TEXT = document.getElementById("comment");
const COMMENTS_LIST = document.getElementById("commentsList");
const PROCESSING_CLASS = "processing";

export function handleCommentPost() {
  if (!POST_COMMENT_BTN.classList.contains(PROCESSING_CLASS)) {
    POST_COMMENT_BTN.classList.add(PROCESSING_CLASS);
    COMMENT_TEXT.classList.add(PROCESSING_CLASS);

    const currentComment = COMMENT_TEXT.innerText;
    const commentTime = new Date();
    const lowercaseSentenceArray = currentComment
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(" ");

    console.log(currentComment);
    const listItemElement = createOnScreeComment(currentComment);
    return {
      lowercaseSentenceArray,
      listItemElement,
      currentComment,
      commentTime,
    };
  }
  console.warn("Already processing");
}

export function handleRemoteComment(data) {
  const comment = data.comment;
  if (comment === undefined) {
    console.warn("Received undefined comment from server");
    return;
  }
  return createOnScreeComment(comment);
}

function createOnScreeComment(currentComment) {
  const currDate = new Date();
  const li = document.createElement("li");
  const p = document.createElement("p");
  const spanName = document.createElement("span");
  const spanDate = document.createElement("span");

  p.innerText = currentComment;

  spanName.setAttribute("class", "username");
  spanName.innerText = "userx";

  spanDate.setAttribute("class", "timestamp");
  spanDate.innerText = currDate.toLocaleString();

  li.appendChild(spanName);
  li.appendChild(spanDate);
  li.appendChild(p);

  COMMENTS_LIST.prepend(li);
  COMMENT_TEXT.innerText = "";

  return li;
}
