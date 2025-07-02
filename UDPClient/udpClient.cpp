/*
Author: Fan Han Hoon
Class:  ECE6122 (Fall 2024)
Last Date Modified: Nov 19,2024
Description: Lab 5: UDP Sockets
What is the purpose of this file?
UDPClient File
*/
#include <SFML/Network.hpp>
#include <iostream>
#include <string>

using namespace sf;

int main(int argc, char* argv[]) {
    // Ensure server IP and port are provided via command-line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <server_ip> <port>\n";
        return 1;
    }

    IpAddress serverIp(argv[1]);
    unsigned short serverPort = static_cast<unsigned short>(std::stoi(argv[2]));

    // Create UDP socket
    UdpSocket socket;

    std::cout << "Enter commands to move the robot (w/s/a/d), change speed (g/h), g to increase by 3 and h to decrease by 3, or quit (q):\n";

    // Input loop for sending commands
    while (true) {
        std::string input;
        std::cout << "> ";
        std::cin >> input;

        // Quit command
        if (input == "q") {
            if (socket.send(input.c_str(), input.size(), serverIp, serverPort) == Socket::Done) {
                std::cout << "Disconnecting from server...\n";
            }
            break;
        }

        // Send user input to server
        if (socket.send(input.c_str(), input.size(), serverIp, serverPort) == Socket::Done) {
            std::cout << "Command sent: " << input << "\n";
        } else {
            std::cerr << "Error: Failed to send command.\n";
        }
    }

    return 0;
}
