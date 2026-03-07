# llm-trace

Single-header C++17 distributed tracing library with RAII spans and OTLP JSON export.

- Single-header C++17, namespace `llm`, stb-style `#ifdef LLM_LLM_TRACE_IMPLEMENTATION` guard
- MIT License — Mattbusel, 2026
- No `/WX` in CMakeLists (avoids C4702 on MSVC)
- Examples use `std::getenv("OPENAI_API_KEY")` for credentials
