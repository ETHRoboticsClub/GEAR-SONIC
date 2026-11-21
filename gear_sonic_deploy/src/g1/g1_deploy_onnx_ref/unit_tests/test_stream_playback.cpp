#include "gtest/gtest.h"
#include "../include/stream_playback.hpp"

TEST(StreamPlayback, IncludesLastCompleteWindow) {
    ASSERT_EQ(NewestCompleteStreamFrame(5, 4), 1);
    // Old >= boundary check incorrectly held frame zero here.
    EXPECT_EQ(AdvanceStreamFrame(0, 5, 4), 1);
    EXPECT_EQ(AdvanceStreamFrame(1, 5, 4), 1);
    EXPECT_EQ(AdvanceStreamFrame(0, 1, 1), 0);
}

TEST(StreamPlayback, WaitsForCompleteContext) {
    EXPECT_FALSE(NewestCompleteStreamFrame(3, 4));
    EXPECT_FALSE(NewestCompleteStreamFrame(0, 4));
    EXPECT_FALSE(NewestCompleteStreamFrame(5, 0));
    EXPECT_EQ(NewestCompleteStreamFrame(4, 4), 0);
    EXPECT_EQ(NewestCompleteStreamFrame(12, 10), 2);
}

TEST(StreamPlayback, DropsBacklogAndHoldsOnMissingPackets) {
    // Start with frames 0..4, then receive a burst through frame 20.
    int cursor = *NewestCompleteStreamFrame(5, 4);
    EXPECT_EQ(cursor, 1);
    cursor = *NewestCompleteStreamFrame(21, 4);
    EXPECT_EQ(cursor, 17);
    // Repeated control ticks without a new packet use the same complete window.
    for (int tick = 0; tick < 10; ++tick)
        EXPECT_EQ(NewestCompleteStreamFrame(21, 4), cursor);
    EXPECT_EQ(NewestCompleteStreamFrame(22, 4), cursor + 1);
}
