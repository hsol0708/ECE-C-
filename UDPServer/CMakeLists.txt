
set(SRCROOT ${PROJECT_SOURCE_DIR}/Lab3/UDPServer)

# all source files
set(SRC ${SRCROOT}/udpServer.cpp)


# define the sockets target
sfml_add_example(UDPServer
                 SOURCES ${SRC})
target_link_libraries(UDPServer sfml-network sfml-graphics sfml-window)
