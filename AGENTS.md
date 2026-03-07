# llm-trace

Single-header C++17 distributed tracing library with RAII spans and OTLP JSON export.

## Structure
- `include/llm_trace.hpp` — single-header implementation
- `examples/` — usage examples
- `CMakeLists.txt` — cmake build (requires vcpkg curl where applicable)

## Build
```bash
cmake -B build && cmake --build build
```
