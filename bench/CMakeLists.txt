file(GLOB ASTER_BENCH_HARNESS CONFIGURE_DEPENDS harness/*.cpp)
add_library(aster_bench_harness ${ASTER_BENCH_HARNESS})
target_include_directories(aster_bench_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aster_bench_harness PUBLIC aster_core)

add_executable(aster-bench aster_bench.cpp)
target_link_libraries(aster-bench PRIVATE aster_bench_harness)
install(TARGETS aster-bench RUNTIME DESTINATION bin)
