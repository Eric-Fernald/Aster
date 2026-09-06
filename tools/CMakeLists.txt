add_executable(aster-bwprobe aster_bwprobe.cpp)
target_link_libraries(aster-bwprobe PRIVATE aster_core)

add_executable(aster-segment-inspect aster_segment_inspect.cpp)
target_link_libraries(aster-segment-inspect PRIVATE aster_core)

add_executable(aster-capabilities aster_capabilities.cpp)
target_link_libraries(aster-capabilities PRIVATE aster_core)

install(TARGETS aster-bwprobe aster-segment-inspect aster-capabilities RUNTIME DESTINATION bin)
