      
// open_spiel/games/mali_ba/mali_ba_common.cc

#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace open_spiel {
namespace mali_ba {

// --- Definitions for the global variables declared in the header ---
bool g_mali_ba_logging_enabled = true;
LogLevel g_mali_ba_log_level = LogLevel::kInfo;
std::string datetime = GetCurrentDateTime(); // Assuming GetCurrentDateTime is in common.h
pid_t pid = getpid(); // Get the current Process ID
std::string g_log_file_path = "/tmp/mali_ba." + datetime + ".pid-" + std::to_string(pid) + ".log";


// --- Definition of the SetLogLevel function ---
void SetLogLevel(LogLevel level) {
    g_mali_ba_log_level = level;
}

// Logging ethod we can call from Python
void LogFromPython(LogLevel level, const std::string& message) {
    // We call the core logging function, but pass "Python" and 0
    // for the file and line, as we don't have that info from the Python side.
    LogMBCore(level, message, true, "Python", 0);
}

// --- Definition of the LogMBCore function ---
// (This should also be moved here if it's not already)
std::string LogLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::kDebug:   return "DEBUG";
        case LogLevel::kInfo:    return "INFO ";
        case LogLevel::kWarning: return "WARN ";
        case LogLevel::kError:   return "ERROR";
        default:                 return "UNKWN";
    }
}

void LogMBCore(LogLevel level, const std::string& message, bool print_to_terminal,
               const char* file, int line) {
    if (!g_mali_ba_logging_enabled || level < g_mali_ba_log_level) {
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::string file_str(file);
    size_t last_slash = file_str.find_last_of('/');
    if (last_slash != std::string::npos) {
        file_str = file_str.substr(last_slash + 1);
    }

    std::stringstream log_stream;
    log_stream << std::put_time(std::localtime(&in_time_t), "%H:%M:%S") << " "
               << "[" << LogLevelToString(level) << "] "
               << "[" << file_str << ":" << line << "] "
               << message;

    if (print_to_terminal) {
        fprintf(stderr, "%s\n", log_stream.str().c_str());
    }
    
    std::ofstream log_file(g_log_file_path, std::ios_base::app);
    if (log_file.is_open()) {
        log_file << log_stream.str() << std::endl;
    }
}

// You might need to move the definitions of PlayerColorToString, etc. here as well
// if they are not directly tied to the State or Game classes.

} // namespace mali_ba
} // namespace open_spiel