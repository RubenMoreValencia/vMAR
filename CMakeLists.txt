cmake_minimum_required(VERSION 3.21)
project(vMAR VERSION 0.1.16 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(VMAR_BUILD_TESTS "Build vMAR numerical contract tests" ON)
option(VMAR_STRICT_PARITY_FIXTURES "Fail configuration when any optional R-C23 fixture is missing" OFF)

# G0.6-D release audit: CTest suite counts depend on fixture availability.
# Strict mode is opt-in for a scientific release gate, not ordinary development.
set(VMAR_OPTIONAL_PARITY_FIXTURES
  vmar_neuron_parity.csv
  vmar_memory_parity.csv
  vmar_control_parity.csv
  vmar_stability_control_audit.csv
  vmar_robustness_parity.csv
  vmar_activation_parity.csv
  vmar_custom_activation_parity.csv
  vmar_adaptive_parity.csv
  vmar_coupling_parity.csv
  vmar_regime_parity.csv
  vmar_intervention_parity.csv
  vmar_transfer_parity.csv
  vmar_adam_parity.csv
  vmar_second_moment_parity.csv)
set(VMAR_FIXTURE_PRESENT 0)
set(VMAR_FIXTURE_MISSING "")
foreach(_vmar_fixture IN LISTS VMAR_OPTIONAL_PARITY_FIXTURES)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/${_vmar_fixture}")
    math(EXPR VMAR_FIXTURE_PRESENT "${VMAR_FIXTURE_PRESENT}+1")
  else()
    list(APPEND VMAR_FIXTURE_MISSING "${_vmar_fixture}")
  endif()
endforeach()
list(LENGTH VMAR_OPTIONAL_PARITY_FIXTURES VMAR_FIXTURE_TOTAL)
message(STATUS "vMAR R-C23 optional fixtures: ${VMAR_FIXTURE_PRESENT}/${VMAR_FIXTURE_TOTAL} available")
if(VMAR_STRICT_PARITY_FIXTURES AND VMAR_FIXTURE_MISSING)
  string(JOIN ", " _vmar_missing_text ${VMAR_FIXTURE_MISSING})
  message(FATAL_ERROR "Missing required parity fixtures (strict mode): ${_vmar_missing_text}")
endif()

add_library(vmar_core STATIC src/vmar_core.c src/vmar_state_layout.c src/vmar_neuron.c src/vmar_memory.c src/vmar_control.c src/vmar_custom_activation.c src/vmar_adaptive_control.c src/vmar_regime.c src/vmar_trace.c src/vmar_network.c src/vmar_network_system.c src/vmar_network_mechanism.c)
add_library(vmar::core ALIAS vmar_core)
target_include_directories(vmar_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(vmar_core PROPERTIES OUTPUT_NAME vmar_core)
set_target_properties(vmar_core PROPERTIES EXPORT_NAME core)
# Static consumers on platforms with a separate math library need the
# transitive dependency; Apple platforms provide these symbols in libSystem.
if(UNIX AND NOT APPLE)
  target_link_libraries(vmar_core PUBLIC m)
endif()

# G0.8-C.1: opt-in research diagnostic. This target and its header are NOT
# installed and do not alter the existing vMAR core ABI or default learning.
option(VMAR_BUILD_EXPERIMENTAL_CURVATURE "Build C01 non-interventional curvature" ON)
if(VMAR_BUILD_EXPERIMENTAL_CURVATURE)
  add_library(vmar_curvature_experimental STATIC src/vmar_curvature.c)
  target_include_directories(vmar_curvature_experimental PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_link_libraries(vmar_curvature_experimental PUBLIC vmar::core)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(vmar_curvature_experimental PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(vmar_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(VMAR_BUILD_TESTS)
  enable_testing()
  if(VMAR_BUILD_EXPERIMENTAL_CURVATURE)
    add_executable(vmar_test_curvature tests/test_curvature.c)
    target_link_libraries(vmar_test_curvature PRIVATE vmar_curvature_experimental m)
    add_test(NAME vmar_curvature_experimental_contracts COMMAND vmar_test_curvature)
    # Explicit R fixture: not silently replaced by a C-generated oracle.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_curvature_parity.csv")
      add_executable(vmar_test_curvature_parity tests/test_curvature_parity.c)
      target_link_libraries(vmar_test_curvature_parity PRIVATE vmar_curvature_experimental m)
      add_test(NAME vmar_r_curvature_parity COMMAND vmar_test_curvature_parity
        ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_curvature_parity.csv)
    endif()
  endif()
  add_executable(vmar_test_core tests/test_core.c)
  target_link_libraries(vmar_test_core PRIVATE vmar::core m)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(vmar_test_core PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME vmar_core_contracts COMMAND vmar_test_core)

  add_executable(vmar_test_state_layout tests/test_state_layout.c)
  target_link_libraries(vmar_test_state_layout PRIVATE vmar::core m)
  add_test(NAME vmar_state_layout_contracts COMMAND vmar_test_state_layout)

  add_executable(vmar_test_formal_contracts tests/test_formal_contracts.c)
  target_link_libraries(vmar_test_formal_contracts PRIVATE vmar::core m)
  add_test(NAME vmar_formal_contract_alignment COMMAND vmar_test_formal_contracts)

  add_executable(vmar_test_trace tests/test_trace.c)
  target_link_libraries(vmar_test_trace PRIVATE vmar::core m)
  add_test(NAME vmar_formalspec_trace_contracts COMMAND vmar_test_trace)

  add_executable(vmar_test_network tests/test_network.c)
  target_link_libraries(vmar_test_network PRIVATE vmar::core m)
  add_test(NAME vmar_multilayer_contracts COMMAND vmar_test_network)

  add_executable(vmar_test_network_system tests/test_network_system.c)
  target_link_libraries(vmar_test_network_system PRIVATE vmar::core m)
  add_test(NAME vmar_multilayer_system_contracts COMMAND vmar_test_network_system)

  add_executable(vmar_test_network_value_audit tests/test_network_value_audit.c)
  target_link_libraries(vmar_test_network_value_audit PRIVATE vmar::core m)
  add_test(NAME vmar_multilayer_value_contracts COMMAND vmar_test_network_value_audit --smoke)

  add_executable(vmar_test_network_mechanism tests/test_network_mechanism.c)
  target_link_libraries(vmar_test_network_mechanism PRIVATE vmar::core m)
  add_test(NAME vmar_multilayer_mechanism_contracts COMMAND vmar_test_network_mechanism)

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_mechanism_parity.csv")
    add_executable(vmar_test_network_mechanism_parity tests/test_network_mechanism_parity.c)
    target_link_libraries(vmar_test_network_mechanism_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_mechanism_parity COMMAND vmar_test_network_mechanism_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_mechanism_parity.csv)
  endif()

  # G0.7-B has its own R oracle; historical strict baseline remains 14/14.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_multilayer_parity.csv")
    add_executable(vmar_test_network_parity tests/test_network_parity.c)
    target_link_libraries(vmar_test_network_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_multilayer_parity COMMAND vmar_test_network_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_multilayer_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_multilayer_system_parity.csv")
    add_executable(vmar_test_network_system_parity tests/test_network_system_parity.c)
    target_link_libraries(vmar_test_network_system_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_multilayer_system_parity COMMAND vmar_test_network_system_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_multilayer_system_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_multilayer_value_audit.csv")
    add_test(NAME vmar_r_multilayer_value_audit COMMAND vmar_test_network_value_audit
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_multilayer_value_audit.csv)
  endif()

  add_executable(vmar_export_trace reference/export_formalspec_trace.c)
  target_link_libraries(vmar_export_trace PRIVATE vmar::core m)
  add_test(NAME vmar_formalspec_trace_export COMMAND vmar_export_trace
    ${CMAKE_CURRENT_BINARY_DIR}/vmar_c3)

  add_executable(vmar_test_parity tests/test_parity.c)
  target_link_libraries(vmar_test_parity PRIVATE vmar::core m)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(vmar_test_parity PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME vmar_r_parity COMMAND vmar_test_parity
    ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_parity.csv)

  add_executable(vmar_test_neuron tests/test_neuron.c)
  target_link_libraries(vmar_test_neuron PRIVATE vmar::core m)
  add_test(NAME vmar_neuron_contracts COMMAND vmar_test_neuron)

  add_executable(vmar_test_activation tests/test_activation.c)
  target_link_libraries(vmar_test_activation PRIVATE vmar::core m)
  add_test(NAME vmar_activation_contracts COMMAND vmar_test_activation)

  add_executable(vmar_test_custom_activation tests/test_custom_activation.c)
  target_link_libraries(vmar_test_custom_activation PRIVATE vmar::core m)
  add_test(NAME vmar_custom_activation_contracts COMMAND vmar_test_custom_activation)

  add_executable(vmar_test_adaptive_control tests/test_adaptive_control.c)
  target_link_libraries(vmar_test_adaptive_control PRIVATE vmar::core m)
  add_test(NAME vmar_adaptive_control_contracts COMMAND vmar_test_adaptive_control)

  add_executable(vmar_test_coupling_invariants tests/test_coupling_invariants.c)
  target_link_libraries(vmar_test_coupling_invariants PRIVATE vmar::core m)
  add_test(NAME vmar_coupling_invariants COMMAND vmar_test_coupling_invariants)

  add_executable(vmar_test_regime tests/test_regime.c)
  target_link_libraries(vmar_test_regime PRIVATE vmar::core m)
  add_test(NAME vmar_regime_contracts COMMAND vmar_test_regime)

  add_executable(vmar_test_intervention_invariants tests/test_intervention_invariants.c)
  target_link_libraries(vmar_test_intervention_invariants PRIVATE vmar::core m)
  add_test(NAME vmar_intervention_invariants COMMAND vmar_test_intervention_invariants)

  add_executable(vmar_test_transfer_invariants tests/test_transfer_invariants.c)
  target_link_libraries(vmar_test_transfer_invariants PRIVATE vmar::core m)
  add_test(NAME vmar_transfer_invariants COMMAND vmar_test_transfer_invariants)

  # Research comparator deliberately excluded from the vmar_core library.
  add_library(vmar_adam_reference STATIC reference/adam_reference.c)
  target_include_directories(vmar_adam_reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vmar_adam_reference PUBLIC vmar::core m)
  add_executable(vmar_test_adam_reference tests/test_adam_reference.c)
  target_link_libraries(vmar_test_adam_reference PRIVATE vmar_adam_reference)
  add_test(NAME vmar_adam_reference_contracts COMMAND vmar_test_adam_reference)

  add_library(vmar_second_moment_reference STATIC reference/second_moment_reference.c)
  target_include_directories(vmar_second_moment_reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vmar_second_moment_reference PUBLIC vmar::core m)
  add_executable(vmar_test_second_moment_reference tests/test_second_moment_reference.c)
  target_link_libraries(vmar_test_second_moment_reference PRIVATE vmar_second_moment_reference)
  add_test(NAME vmar_second_moment_contracts COMMAND vmar_test_second_moment_reference)

  add_executable(vmar_test_memory tests/test_memory.c)
  target_link_libraries(vmar_test_memory PRIVATE vmar::core m)
  add_test(NAME vmar_memory_contracts COMMAND vmar_test_memory)

  add_executable(vmar_test_control tests/test_control.c)
  target_link_libraries(vmar_test_control PRIVATE vmar::core m)
  add_test(NAME vmar_control_contracts COMMAND vmar_test_control)

  add_executable(vmar_test_audit_invariants tests/test_audit_invariants.c)
  target_link_libraries(vmar_test_audit_invariants PRIVATE vmar::core m)
  add_test(NAME vmar_stability_audit_invariants COMMAND vmar_test_audit_invariants)

  add_executable(vmar_test_robustness tests/test_robustness.c)
  target_link_libraries(vmar_test_robustness PRIVATE vmar::core m)
  add_test(NAME vmar_robustness_contracts COMMAND vmar_test_robustness)

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_neuron_parity.csv")
    add_executable(vmar_test_neuron_parity tests/test_neuron_parity.c)
    target_link_libraries(vmar_test_neuron_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_neuron_parity COMMAND vmar_test_neuron_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_neuron_parity.csv)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_memory_parity.csv")
    add_executable(vmar_test_memory_parity tests/test_memory_parity.c)
    target_link_libraries(vmar_test_memory_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_memory_parity COMMAND vmar_test_memory_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_memory_parity.csv)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_control_parity.csv")
    add_executable(vmar_test_control_parity tests/test_control_parity.c)
    target_link_libraries(vmar_test_control_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_control_parity COMMAND vmar_test_control_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_control_parity.csv)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_stability_control_audit.csv")
    add_executable(vmar_test_audit_parity tests/test_audit_parity.c)
    target_link_libraries(vmar_test_audit_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_stability_audit COMMAND vmar_test_audit_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_stability_control_audit.csv)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_robustness_parity.csv")
    add_executable(vmar_test_robustness_parity tests/test_robustness_parity.c)
    target_link_libraries(vmar_test_robustness_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_robustness_parity COMMAND vmar_test_robustness_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_robustness_parity.csv)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_activation_parity.csv")
    add_executable(vmar_test_activation_parity tests/test_neuron_parity.c)
    target_compile_definitions(vmar_test_activation_parity PRIVATE VMAR_EXPECT_CASES=48)
    target_link_libraries(vmar_test_activation_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_activation_parity COMMAND vmar_test_activation_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_activation_parity.csv)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_custom_activation_parity.csv")
    add_executable(vmar_test_custom_activation_parity tests/test_custom_activation_parity.c)
    target_link_libraries(vmar_test_custom_activation_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_custom_activation_parity COMMAND vmar_test_custom_activation_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_custom_activation_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_adaptive_parity.csv")
    add_executable(vmar_test_adaptive_parity tests/test_adaptive_parity.c)
    target_link_libraries(vmar_test_adaptive_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_adaptive_parity COMMAND vmar_test_adaptive_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_adaptive_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_coupling_parity.csv")
    add_executable(vmar_test_coupling_parity tests/test_coupling_parity.c)
    target_link_libraries(vmar_test_coupling_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_coupling_parity COMMAND vmar_test_coupling_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_coupling_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_regime_parity.csv")
    add_executable(vmar_test_regime_parity tests/test_regime_parity.c)
    target_link_libraries(vmar_test_regime_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_regime_parity COMMAND vmar_test_regime_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_regime_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_intervention_parity.csv")
    add_executable(vmar_test_intervention_parity tests/test_intervention_parity.c)
    target_link_libraries(vmar_test_intervention_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_intervention_parity COMMAND vmar_test_intervention_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_intervention_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_transfer_parity.csv")
    add_executable(vmar_test_transfer_parity tests/test_transfer_parity.c)
    target_link_libraries(vmar_test_transfer_parity PRIVATE vmar::core m)
    add_test(NAME vmar_r_transfer_parity COMMAND vmar_test_transfer_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_transfer_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_adam_parity.csv")
    add_executable(vmar_test_adam_parity tests/test_adam_parity.c)
    target_link_libraries(vmar_test_adam_parity PRIVATE vmar_adam_reference)
    add_test(NAME vmar_r_adam_reference_parity COMMAND vmar_test_adam_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_adam_parity.csv)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_second_moment_parity.csv")
    add_executable(vmar_test_second_moment_parity tests/test_second_moment_parity.c)
    target_link_libraries(vmar_test_second_moment_parity PRIVATE vmar_second_moment_reference)
    add_test(NAME vmar_r_second_moment_parity COMMAND vmar_test_second_moment_parity
      ${CMAKE_CURRENT_SOURCE_DIR}/reference/vmar_second_moment_parity.csv)
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS vmar_core EXPORT vMARCoreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/vmar/vmar.h include/vmar/state_layout.h include/vmar/neuron.h include/vmar/memory.h include/vmar/control.h include/vmar/custom_activation.h include/vmar/adaptive_control.h include/vmar/regime.h include/vmar/trace.h include/vmar/network.h include/vmar/network_system.h include/vmar/network_mechanism.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vmar)
install(EXPORT vMARCoreTargets NAMESPACE vmar::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vMAR)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vMARConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/vMARConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vMAR)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/vMARConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/vMARConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/vMARConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vMAR)
