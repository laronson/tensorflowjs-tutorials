const http = require("http");
const express = require("express");
const app = express();
const server = http.createServer(app);

var io = require("socket.io")(server);

io.on("connect", (socket) => {
  console.log("client connected");
  socket.on("comment", (data) => {
    socket.broadcast.emit("remoteComment", data);
  });
});

// Make all the files in 'www' available.
app.use(express.static("www"));

app.get("/", (request, response) => {
  response.sendFile(__dirname + "/www/index.html");
});

// Listen for requests.
const listener = server.listen(process.env.PORT, () => {
  console.log("Your app is listening on port " + listener.address().port);
});
