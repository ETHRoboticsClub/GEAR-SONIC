#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>

// A level-triggered wakeup, not a packet queue. Notifications arriving during
// preparation are retained; bursts collapse into one pass over the latest data.
class InputWakeup {
 public:
    void Notify() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_ = true;
        }
        cv_.notify_one();
    }

    // true: process input (packet or keyboard fallback); false: shut down.
    bool Wait(std::chrono::steady_clock::duration fallback) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, fallback, [this] { return pending_ || stopped_; });
        pending_ = false;
        return !stopped_;
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

 private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool pending_ = false;
    bool stopped_ = false;
};
