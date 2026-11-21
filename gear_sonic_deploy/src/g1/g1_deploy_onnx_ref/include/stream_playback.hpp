#pragma once

#include <algorithm>
#include <optional>

// A horizon includes the current frame. Four samples therefore require
// three later frames, and the last valid start is frame_count - 4.
inline std::optional<int> NewestCompleteStreamFrame(int frame_count, int horizon) {
    if (horizon < 1 || frame_count < horizon) return std::nullopt;
    return frame_count - horizon;
}

inline int AdvanceStreamFrame(int current_frame, int frame_count, int horizon) {
    const auto last = NewestCompleteStreamFrame(frame_count, std::max(1, horizon));
    return std::clamp(current_frame + 1, 0, last.value_or(0));
}
