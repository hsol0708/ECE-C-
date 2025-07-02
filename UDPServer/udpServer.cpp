/*
Author: Fan Han Hoon
Class:  ECE6122 (Fall 2024)
Last Date Modified: Nov 19,2024
Description: Lab 5: UDP Sockets
What is the purpose of this file?
UDPServer File
*/
#include <SFML/Network.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>

using namespace sf;

int main(int argc, char* argv[]) {
    // Ensure a port number is provided via command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <port>\n";
        return 1;
    }

    unsigned short port = static_cast<unsigned short>(std::stoi(argv[1]));

    // Create and bind the UDP socket
    UdpSocket socket;
    if (socket.bind(port) != Socket::Done) {
        std::cerr << "Error: Unable to bind socket to port " << port << ".\n";
        return 1;
    }
    std::cout << "Server is listening on port " << port << "...\n";

    // Create SFML window for rendering
    RenderWindow window(VideoMode(800, 600), "UDP Robot Server");
    CircleShape robot(10);  // Robot represented as a circle
    robot.setFillColor(Color::Green);
    robot.setPosition(395, 295);  // Initial position

    float speed = 3.0f;  // Initial speed
    bool robotVisible = true;
    std::string receivedCommand;

    // Main loop
    while (window.isOpen()) {
        // Handle window events
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed) {
                // Notify clients of shutdown and close window
                const std::string shutdownMsg = "server_shutdown";
                if (socket.send(shutdownMsg.c_str(), shutdownMsg.size(), IpAddress::Broadcast, port) != Socket::Done) {
                    std::cerr << "Failed to notify clients about shutdown.\n";
                }
                std::cout << "Server is shutting down.\n";
                window.close();
            }
        }

        // Receive and process commands
        char data[100];
        std::size_t received;
        IpAddress sender;
        unsigned short senderPort;
        if (socket.receive(data, sizeof(data), received, sender, senderPort) == Socket::Done) {
            receivedCommand = std::string(data, received);
            std::cout << "Received command: " << receivedCommand << " from " << sender << ":" << senderPort << "\n";

            // Update robot's position based on received command
            if (receivedCommand == "w") robot.move(0, -speed);  // Move up
            if (receivedCommand == "s") robot.move(0, speed);   // Move down
            if (receivedCommand == "a") robot.move(-speed, 0);  // Move left
            if (receivedCommand == "d") robot.move(speed, 0);   // Move right
            if (receivedCommand == "g") speed += 1.0f;          // Increase speed
            if (receivedCommand == "h" && speed > 1.0f) speed -= 1.0f;  // Decrease speed
            if (receivedCommand == "q") {  // Client disconnect
                robotVisible = false;
                std::cout << "Client disconnected.\n";
                robot.setPosition(395, 295);  // Reset 
            }
        }

        // Render the robot
        window.clear(Color::Black);
        if (robotVisible) {
            window.draw(robot);
        }
        else{
            window.close();
        }
        window.draw(robot);
        window.display();
    }

    return 0;
}
