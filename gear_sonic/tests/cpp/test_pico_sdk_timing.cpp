// Exercise the real SDK JSON callback without initializing a headset connection.
#include <cassert>
#include <cstring>
#include "../../../external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/bindings/py_bindings.cpp"

int main() {
    json body;
    body["joints"] = json::array();
    for (int joint = 0; joint < 24; ++joint)
        body["joints"].push_back({{"p", "0,1,0,0,0,0,1"}});
    json value = {{"Controller", json::object()}, {"Hand", json::object()},
                  {"Body", body}, {"timeStampNs", int64_t{1700000000000000000}}};
    PXREADevStateJson state{};
    auto submit = [&] {
        const std::string wire = json({{"value", value.dump()}}).dump();
        assert(wire.size() < sizeof(state.stateJson));
        std::memcpy(state.stateJson, wire.c_str(), wire.size() + 1);
        OnPXREAClientCallback(nullptr, PXREADeviceStateJson, 0, &state);
    };
    const auto before = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    submit();
    auto first = getBodyInputTiming();
    assert(first.at("body_sequence") == 1);
    assert(first.at("headset_frame_unix_ns") == 1700000000000000000LL);
    assert(first.at("sdk_receipt_monotonic_ns") >= before);
    assert(first.at("sdk_receipt_unix_ns") > 1700000000000000000LL);
    value["timeStampNs"] = int64_t{1700000000011000000};
    submit();  // Identical body poses still have a new message timestamp/counter.
    auto second = getBodyInputTiming();
    assert(second.at("body_sequence") == 2);
    assert(second.at("headset_frame_unix_ns") == 1700000000011000000LL);
    assert(second.at("sdk_receipt_monotonic_ns") >= first.at("sdk_receipt_monotonic_ns"));
    value.erase("timeStampNs");
    submit();
    assert(getBodyInputTiming().at("headset_frame_unix_ns") == 0);
    std::cout << "SDK body timing callback tests passed\n";
}
