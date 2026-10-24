// Process CPU time (user+kernel) so timings stay meaningful on a loaded shared machine.
#pragma once
#ifdef _WIN32
#include <windows.h>
namespace cs { namespace i3 {
inline double cpu_ms() {
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    auto t = [](FILETIME f) { return (double)(((unsigned long long)f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e4; };
    return t(k) + t(u);
}
}}
#else
#include <ctime>
namespace cs { namespace i3 { inline double cpu_ms() { return 1000.0 * (double)clock() / CLOCKS_PER_SEC; } }}
#endif
