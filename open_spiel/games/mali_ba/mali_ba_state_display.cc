// mali_ba_state_display.cc
// State display and rendering functionality

#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/hex_grid.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <string>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h" // Use for formatted strings
#include "open_spiel/abseil-cpp/absl/strings/string_view.h" // Include for string_view

namespace open_spiel {
namespace mali_ba {

// Helper struct for board display
struct RowDisplay {
    std::string left_side;
    std::string right_side;
};

// A safe implementation of ToString() that avoids std::stringstream.
std::string Mali_BaState::ToString() const {
    std::string out; // Build the output string directly.

    // Handle special cases
    if (IsChanceNode()) {
        return "Chance Node Setup Phase\n";
    }
    
    // Player info header
    absl::StrAppend(&out, "Current Player: ", PlayerColorToString(current_player_color_),
                    " (ID: ", current_player_id_, ")\n\n");

    if (GetGame()->GetValidHexes().empty()) {
        absl::StrAppend(&out, "[Board Empty]");
        return out;
    }

    // --- Pre-calculate board bounds and EXTENTS ---
    int global_min_y = 0, global_max_y = 0;
    int global_left_extent = 0;
    std::map<int, std::pair<int, int>> y_to_x_range;
    std::map<int, int> row_max_left_extent;
    bool first_hex = true;

    for (const auto& hex : GetGame()->GetValidHexes()) {
        if (first_hex) {
            global_min_y = global_max_y = hex.y;
            y_to_x_range[hex.y] = {hex.x, hex.x};
            first_hex = false;
        } else {
            global_min_y = std::min(global_min_y, hex.y);
            global_max_y = std::max(global_max_y, hex.y);
            if (y_to_x_range.find(hex.y) == y_to_x_range.end()) {
                y_to_x_range[hex.y] = {hex.x, hex.x};
            } else {
                y_to_x_range[hex.y].first = std::min(y_to_x_range[hex.y].first, hex.x);
                y_to_x_range[hex.y].second = std::max(y_to_x_range[hex.y].second, hex.x);
            }
        }
        int extent = std::abs(hex.x) + std::abs(hex.z);
        if (hex.x <= 0) {
            global_left_extent = std::max(global_left_extent, extent);
            row_max_left_extent[hex.y] = std::max(row_max_left_extent[hex.y], extent);
        }
    }

    const int hex_print_width = 14;
    const int indent_multiplier = hex_print_width / 2;
    const int content_padding_width = 1;

    // --- format_coord lambda ---
    auto format_coord = [&](const HexCoord& hex) -> std::string {
        std::string formatted_str = absl::StrFormat("[%3d,%3d,%3d]", hex.x, hex.y, hex.z);
        formatted_str.resize(hex_print_width, ' ');
        return formatted_str;
    };

    // --- format_content lambda ---
    auto format_content = [this](const HexCoord& hex) -> std::string {
        std::string content_str;
        std::string prefix_str;
        std::string summary_str;
        std::string details_str;

        int city_id = -1;
        for (const auto& city : GetGame()->GetCities()) {
            if (city.location == hex) {
                city_id = city.id;
                break;
            }
        }
        if (city_id != -1) {
            prefix_str = absl::StrCat("C", city_id);
        }

        PlayerColor token_owner = GetPlayerTokenAt(hex);
        if (token_owner != PlayerColor::kEmpty) {
            // FIX: Don't pass a single char. Pass a string_view or string literal.
            char c = PlayerColorToChar(token_owner);
            absl::StrAppend(&summary_str, absl::string_view(&c, 1));
        }
        
        const auto& posts = GetTradePostsAt(hex);
        for (const auto& post : posts) {
            if (post.type != TradePostType::kNone) {
                // FIX: Same here.
                char owner_char = PlayerColorToChar(post.owner);
                char type_char = (post.type == TradePostType::kPost ? 'p' : 'T');
                absl::StrAppend(&summary_str, absl::string_view(&owner_char, 1), absl::string_view(&type_char, 1));
            }
        }

        const auto& meeples = GetMeeplesAt(hex);
        if (!meeples.empty()) {
            std::map<MeepleColor, int> meeple_color_counts;
            for (MeepleColor mc : meeples) meeple_color_counts[mc]++;
            
            std::vector<std::string> detail_parts;
            std::vector<std::pair<MeepleColor, int>> sorted_counts(meeple_color_counts.begin(), meeple_color_counts.end());
            std::sort(sorted_counts.begin(), sorted_counts.end());

            for (const auto& [mc, count] : sorted_counts) {
                if (count > 0) {
                    detail_parts.push_back(absl::StrCat(MeepleColorToString(mc), count));
                }
            }
            if (!detail_parts.empty()) {
                details_str = absl::StrCat("m", meeples.size(), ":", absl::StrJoin(detail_parts, ","));
            } else {
                details_str = absl::StrCat("m", meeples.size());
            }
        }

        content_str += prefix_str;
        if (!summary_str.empty()) {
            if (!prefix_str.empty()) content_str += "; ";
            content_str += summary_str;
        }
        if (!details_str.empty()) {
            if (!prefix_str.empty() || !summary_str.empty()) {
                content_str += (summary_str.empty() ? ":" : " ");
            }
            content_str += details_str;
        }

        return content_str.empty() ? "(.)" : absl::StrCat("(", content_str, ")");
    };

    // --- Build rows ---
    std::vector<RowDisplay> rows;
    int max_left_width = 0;
    for (int y = global_max_y; y >= global_min_y; --y) {
        if (y_to_x_range.find(y) == y_to_x_range.end()) continue;

        RowDisplay current_row_display;
        int current_row_left_extent = row_max_left_extent[y];
        int indent = std::max(0, (global_left_extent - current_row_left_extent) * indent_multiplier);
        current_row_display.left_side.append(indent, ' ');

        int row_min_x = y_to_x_range[y].first;
        int row_max_x = y_to_x_range[y].second;
        bool first_valid_hex_in_row = true;

        for (int x = row_min_x; x <= row_max_x; ++x) {
            HexCoord hex(x, y, -x - y);
            if (IsValidHex(hex)) {
                absl::StrAppend(&current_row_display.left_side, format_coord(hex));
                std::string content = format_content(hex);
                if (!first_valid_hex_in_row) {
                    current_row_display.right_side.append(content_padding_width, ' ');
                }
                absl::StrAppend(&current_row_display.right_side, content);
                first_valid_hex_in_row = false;
            }
        }
        rows.push_back(current_row_display);
        max_left_width = std::max(max_left_width, static_cast<int>(current_row_display.left_side.length()));
    }

    // --- Output rows to the main string `out` ---
    int separator_pos = max_left_width + 2;
    for (const auto& row_display : rows) {
        absl::StrAppend(&out, row_display.left_side);
        int spaces_needed = std::max(0, separator_pos - static_cast<int>(row_display.left_side.length()));
        out.append(spaces_needed, ' ');
        absl::StrAppend(&out, "| ", row_display.right_side, "\n");
    }

    // --- Resources ---
    absl::StrAppend(&out, "\n--- Resources ---\n");
    for (Player p = 0; p < game_->NumPlayers(); ++p) {
        PlayerColor pc = GetPlayerColor(p);
        absl::StrAppend(&out, PlayerColorToString(pc), " (ID:", p, "):\n");
        
        std::vector<std::string> common_items;
        // Use a map to sort goods for consistent output
        std::map<std::string, int> sorted_common(GetPlayerCommonGoods(p).begin(), GetPlayerCommonGoods(p).end());
        for (const auto& [name, count] : sorted_common) {
            common_items.push_back(absl::StrCat(name, ":", count));
        }
        absl::StrAppend(&out, "  Common: {", absl::StrJoin(common_items, ", "), "}\n");
        
        std::vector<std::string> rare_items;
        std::map<std::string, int> sorted_rare(GetPlayerRareGoods(p).begin(), GetPlayerRareGoods(p).end());
        for (const auto& [name, count] : sorted_rare) {
            rare_items.push_back(absl::StrCat(name, ":", count));
        }
        absl::StrAppend(&out, "  Rare:   {", absl::StrJoin(rare_items, ", "), "}\n");
    }
    absl::StrAppend(&out, "---------------\n");

    return out;
}

}  // namespace mali_ba
}  // namespace open_spiel