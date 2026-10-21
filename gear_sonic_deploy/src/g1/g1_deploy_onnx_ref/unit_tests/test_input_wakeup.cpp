#include "gtest/gtest.h"
#include "../include/input_wakeup.hpp"
#include <future>
#include <thread>

using namespace std::chrono_literals;

TEST(InputWakeup, PacketWakesWaitingWorkerWithoutPollingTimeout) {
    InputWakeup signal;
    auto result = std::async(std::launch::async, [&] { return signal.Wait(5s); });
    signal.Notify(); // Also covers notification before the worker enters Wait.
    const auto status = result.wait_for(1s);
    if (status != std::future_status::ready) signal.Stop();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_TRUE(result.get());
}

TEST(InputWakeup, BurstDuringPreparationIsRetainedAndCoalesced) {
    InputWakeup signal;
    for (int i = 0; i < 100; ++i) signal.Notify();
    auto first = std::async(std::launch::async, [&] { return signal.Wait(5s); });
    const auto status = first.wait_for(1s);
    if (status != std::future_status::ready) signal.Stop();
    ASSERT_EQ(status, std::future_status::ready);
    ASSERT_TRUE(first.get());
    // No queue of 99 pending packets: the next wait reaches the idle timeout.
    const auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(signal.Wait(20ms));
    EXPECT_GE(std::chrono::steady_clock::now() - start, 15ms);
    signal.Notify(); // A packet during work must survive until the next Wait.
    auto next = std::async(std::launch::async, [&] { return signal.Wait(5s); });
    const auto next_status = next.wait_for(1s);
    if (next_status != std::future_status::ready) signal.Stop();
    EXPECT_EQ(next_status, std::future_status::ready);
    EXPECT_TRUE(next.get());
}

TEST(InputWakeup, ShutdownUnblocksWorkerAndOverridesPendingPackets) {
    InputWakeup signal;
    auto result = std::async(std::launch::async, [&] { return signal.Wait(5s); });
    signal.Stop();
    EXPECT_EQ(result.wait_for(1s), std::future_status::ready);
    EXPECT_FALSE(result.get());
    signal.Notify();
    EXPECT_FALSE(signal.Wait(5s));
}
