const io = require("socket.io"), server = io.listen(8069);

server.on("connection", (socket) => {
    console.info(`Connection with id=${socket.id}`);
    	socket.on("*", (event) => {
		console.log(event);
	})
	socket.on("disconnect", () => {
        console.info(`Disconnected id=${socket.id}`);
    });
}, 1000);