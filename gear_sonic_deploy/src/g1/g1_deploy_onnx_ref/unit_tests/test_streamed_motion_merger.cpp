#include "gtest/gtest.h"
#include "../include/motion_data_reader.hpp"
#include "../include/input_interface/streamed_motion_merger.hpp"
#include "../include/stream_playback.hpp"

namespace {

// A pose chunk whose root quaternion's first component carries the sender's
// frame index, so the merged window shows which frames it holds.
StreamedMotionMerger::IncomingData Chunk(int first_index, int frames = 5) {
    StreamedMotionMerger::IncomingData data;
    data.protocol_version = 2;
    data.num_frames = frames;
    data.num_quat_bodies = 1;
    data.num_smpl_joints = 24;
    data.num_smpl_poses = 21;
    for (int i = 0; i < frames; ++i) {
        data.frame_indices.push_back(first_index + i);
        data.body_quat.push_back({{static_cast<double>(first_index + i), 0.0, 0.0, 0.0}});
        data.smpl_joints.emplace_back(24, std::array<double, 3>{0.0, 0.0, 0.0});
        data.smpl_pose.emplace_back(21, std::array<double, 3>{0.0, 0.0, 0.0});
    }
    return data;
}

int FrameIndexAt(const MotionSequence& motion, int frame) {
    return static_cast<int>(motion.BodyQuaternions(frame)[0][0]);
}

// The cursor the controller uses with --stream-lookahead-frames 4.
int LatestWindow(const MotionSequence& motion) {
    return *NewestCompleteStreamFrame(motion.timesteps, 4);
}

}  // namespace

TEST(StreamedMotionMerger, OverlappingChunksExtendTheWindow) {
    StreamedMotionMerger merger;
    auto result = merger.MergeIncomingData(Chunk(0), 0);
    EXPECT_TRUE(result.did_catchup_reset);  // The first chunk starts the stream.
    int cursor = LatestWindow(*result.motion);
    for (int first = 1; first < 40; ++first) {
        result = merger.MergeIncomingData(Chunk(first), cursor);
        ASSERT_FALSE(result.did_catchup_reset);
        EXPECT_EQ(result.bridged_gap_frames, 0);
        cursor = LatestWindow(*result.motion);
        EXPECT_EQ(FrameIndexAt(*result.motion, cursor), first + 1);
        EXPECT_EQ(FrameIndexAt(*result.motion, result.motion->timesteps - 1), first + 4);
    }
}

TEST(StreamedMotionMerger, ShortTrackingGapKeepsPlaybackAndHeading) {
    StreamedMotionMerger merger;
    auto result = merger.MergeIncomingData(Chunk(0), 0);
    int cursor = LatestWindow(*result.motion);
    for (int first = 1; first <= 20; ++first) {
        result = merger.MergeIncomingData(Chunk(first), cursor);
        cursor = LatestWindow(*result.motion);
    }
    // Tracking stalls for 160 ms: the sender skips frames 25..32.
    result = merger.MergeIncomingData(Chunk(33), cursor);
    EXPECT_FALSE(result.did_catchup_reset);  // No heading re-alignment.
    EXPECT_EQ(result.bridged_gap_frames, 8);
    cursor = LatestWindow(*result.motion);
    // The newest window is entirely after the gap, in order.
    for (int i = 0; i < 4; ++i) EXPECT_EQ(FrameIndexAt(*result.motion, cursor + i), 34 + i);
    // Frames before the gap stay in the window as history.
    EXPECT_EQ(FrameIndexAt(*result.motion, cursor - 2), 24);
    // The stream then continues as if nothing had been skipped.
    for (int first = 34; first < 60; ++first) {
        result = merger.MergeIncomingData(Chunk(first), cursor);
        ASSERT_FALSE(result.did_catchup_reset);
        EXPECT_EQ(result.bridged_gap_frames, 0);
        cursor = LatestWindow(*result.motion);
        EXPECT_EQ(FrameIndexAt(*result.motion, result.motion->timesteps - 1), first + 4);
    }
}

TEST(StreamedMotionMerger, LongDropoutRestartsTheStream) {
    StreamedMotionMerger merger;
    auto result = merger.MergeIncomingData(Chunk(0), 0);
    result = merger.MergeIncomingData(Chunk(1), LatestWindow(*result.motion));
    const int skipped = StreamedMotionMerger::MAX_BRIDGED_GAP_FRAMES + 1;
    result = merger.MergeIncomingData(Chunk(6 + skipped), LatestWindow(*result.motion));
    EXPECT_TRUE(result.did_catchup_reset);
    EXPECT_EQ(result.bridged_gap_frames, 0);
    EXPECT_EQ(result.motion->timesteps, 5);
    EXPECT_EQ(result.window_start, 6 + skipped);
}

TEST(StreamedMotionMerger, RestartedSenderResetsAfterABridgedGap) {
    StreamedMotionMerger merger;
    auto result = merger.MergeIncomingData(Chunk(0), 0);
    result = merger.MergeIncomingData(Chunk(15), LatestWindow(*result.motion));
    ASSERT_EQ(result.bridged_gap_frames, 10);
    // Leaving and re-entering POSE mode restarts the sender's indices at zero.
    result = merger.MergeIncomingData(Chunk(0), LatestWindow(*result.motion));
    EXPECT_TRUE(result.did_catchup_reset);
    EXPECT_EQ(result.window_start, 0);
    result = merger.MergeIncomingData(Chunk(1), LatestWindow(*result.motion));
    EXPECT_FALSE(result.did_catchup_reset);
    EXPECT_EQ(result.bridged_gap_frames, 0);
    EXPECT_EQ(FrameIndexAt(*result.motion, result.motion->timesteps - 1), 5);
}
