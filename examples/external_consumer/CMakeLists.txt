cmake_minimum_required(VERSION 3.21)
project(vmar_external_consumer LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(vMAR CONFIG REQUIRED)

add_executable(vmar_external_consumer main.c)
target_link_libraries(vmar_external_consumer PRIVATE vmar::core)

enable_testing()
add_test(NAME vmar_external_consumer_smoke COMMAND vmar_external_consumer)
