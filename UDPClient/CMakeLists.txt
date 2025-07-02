
set(SRCROOT ${PROJECT_SOURCE_DIR}/Lab3/UDPClient)

# all source files
set(SRC ${SRCROOT}/udpClient.cpp)

# define the sockets target
sfml_add_example(UDPClient
                 SOURCES ${SRC})
target_link_libraries(UDPClient sfml-network sfml-graphics sfml-window)
