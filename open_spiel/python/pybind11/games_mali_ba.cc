#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/hex_grid.h"
#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

namespace open_spiel {

void init_pyspiel_games_mali_ba(::pybind11::module &m) {
    // Create a submodule
    py::module_ mali_ba = m.def_submodule("mali_ba");
    
    // Define constants (Fixed to match mali_ba_common.h)
    m.attr("INVALID_ACTION") = py::int_(mali_ba::kInvalidAction); 
    m.attr("PASS_ACTION") = py::int_(mali_ba::kPassAction);
    m.attr("CHANCE_SETUP_ACTION") = py::int_(mali_ba::kChanceSetupAction); 
    m.attr("MAX_ACTIONS") = py::int_(mali_ba::kMaxActions); 
    
    // Enums
    py::enum_<open_spiel::mali_ba::Phase>(mali_ba, "Phase")
        .value("EMPTY", open_spiel::mali_ba::Phase::kEmpty)
        .value("SETUP", open_spiel::mali_ba::Phase::kSetup)
        .value("PLACE_TOKEN", open_spiel::mali_ba::Phase::kPlaceToken)
        .value("PLAY", open_spiel::mali_ba::Phase::kPlay)
        .value("MANCALA_STEP", open_spiel::mali_ba::Phase::kMancalaStep)
        .value("MANCALA_TOKEN_STEP", open_spiel::mali_ba::Phase::kMancalaTokenStep)
        .value("OPTIONAL_POST", open_spiel::mali_ba::Phase::kOptionalPost)
        .value("OPTIONAL_POST_PAYMENT", open_spiel::mali_ba::Phase::kOptionalPostPayment)
        .value("OPTIONAL_ROUTE", open_spiel::mali_ba::Phase::kOptionalRoute)
        .value("END_ROUND", open_spiel::mali_ba::Phase::kEndRound)
        .value("GAME_OVER", open_spiel::mali_ba::Phase::kGameOver)
        .export_values();
        
    py::enum_<open_spiel::mali_ba::PlayerColor>(mali_ba, "Mali_BaPlayerColor")
        .value("EMPTY", open_spiel::mali_ba::PlayerColor::kEmpty)
        .value("RED", open_spiel::mali_ba::PlayerColor::kRed)
        .value("GREEN", open_spiel::mali_ba::PlayerColor::kGreen)
        .value("BLUE", open_spiel::mali_ba::PlayerColor::kBlue)
        .value("VIOLET", open_spiel::mali_ba::PlayerColor::kViolet)
        .value("PINK", open_spiel::mali_ba::PlayerColor::kPink)
        .export_values();
        
    // Basic classes
    py::class_<open_spiel::mali_ba::HexCoord>(mali_ba, "HexCoord")
        .def(py::init<int, int, int>())
        .def_readonly("x", &open_spiel::mali_ba::HexCoord::x)
        .def_readonly("y", &open_spiel::mali_ba::HexCoord::y)
        .def_readonly("z", &open_spiel::mali_ba::HexCoord::z)
        .def("__str__", &open_spiel::mali_ba::HexCoord::ToString);
        
    py::class_<open_spiel::mali_ba::TradePost>(mali_ba, "TradePost")
        .def(py::init<>())
        .def_readonly("owner", &open_spiel::mali_ba::TradePost::owner)
        .def_readonly("type", &open_spiel::mali_ba::TradePost::type);

    py::class_<open_spiel::mali_ba::City>(mali_ba, "City")
        .def(py::init<>())
        .def_readonly("id", &open_spiel::mali_ba::City::id)
        .def_readonly("name", &open_spiel::mali_ba::City::name)
        .def_readonly("culture", &open_spiel::mali_ba::City::culture)
        .def_readonly("location", &open_spiel::mali_ba::City::location)
        .def_readonly("common_good", &open_spiel::mali_ba::City::common_good)
        .def_readonly("rare_good", &open_spiel::mali_ba::City::rare_good);

    py::class_<mali_ba::TradeRoute>(mali_ba, "TradeRoute")
        .def(py::init<>())
        .def_readonly("id", &mali_ba::TradeRoute::id)
        .def_readonly("owner", &mali_ba::TradeRoute::owner)
        .def_readonly("hexes", &mali_ba::TradeRoute::hexes)
        .def_readonly("goods", &mali_ba::TradeRoute::goods)
        .def_readonly("active", &mali_ba::TradeRoute::active);

    // TrainingParameters struct
    py::class_<mali_ba::TrainingParameters>(mali_ba, "TrainingParameters")
        .def_readonly("time_penalty", &mali_ba::TrainingParameters::time_penalty)
        .def_readonly("max_moves_penalty", &mali_ba::TrainingParameters::max_moves_penalty)
        .def_readonly("draw_penalty", &mali_ba::TrainingParameters::draw_penalty)
        .def_readonly("loss_penalty", &mali_ba::TrainingParameters::loss_penalty)
        .def_readonly("upgrade_reward", &mali_ba::TrainingParameters::upgrade_reward)
        .def_readonly("trade_route_reward", &mali_ba::TrainingParameters::trade_route_reward)
        .def_readonly("new_rare_region_reward", &mali_ba::TrainingParameters::new_rare_region_reward)
        .def_readonly("new_common_good_reward", &mali_ba::TrainingParameters::new_common_good_reward)
        .def_readonly("key_location_post_reward", &mali_ba::TrainingParameters::key_location_post_reward)
        .def_readonly("quick_win_bonus", &mali_ba::TrainingParameters::quick_win_bonus)
        .def_readonly("quick_win_threshold", &mali_ba::TrainingParameters::quick_win_threshold);

    // State class - use py::classh to match base State registration in pyspiel.cc
    py::classh<mali_ba::Mali_BaState, open_spiel::State> state_class_binder(m, "Mali_BaState");
    state_class_binder
        .def("play_random_move_and_serialize", &mali_ba::Mali_BaState::PlayRandomMoveAndSerialize)
        .def("select_heuristic_random_action", &mali_ba::Mali_BaState::SelectHeuristicRandomAction)
        .def("get_player_common_goods", &mali_ba::Mali_BaState::GetPlayerCommonGoods, py::return_value_policy::reference_internal)
        .def("get_player_rare_goods", &mali_ba::Mali_BaState::GetPlayerRareGoods, py::return_value_policy::reference_internal)
        .def("parse_move_string_to_action", &mali_ba::Mali_BaState::ParseMoveStringToAction)
        .def("create_trade_route", &mali_ba::Mali_BaState::CreateTradeRoute)
        .def("delete_trade_route", &mali_ba::Mali_BaState::DeleteTradeRoute)
        .def("validate_trade_routes", &mali_ba::Mali_BaState::ValidateTradeRoutes)
        .def("apply_income_collection", &mali_ba::Mali_BaState::ApplyIncomeCollection)
        .def("serialize", &mali_ba::Mali_BaState::Serialize)
        .def("create_setup_json", &mali_ba::Mali_BaState::CreateSetupJson)
        .def("get_heuristic_action_weights", &mali_ba::Mali_BaState::GetHeuristicActionWeights)
        .def("get_game_end_reason", &mali_ba::Mali_BaState::GetGameEndReason)
        .def("get_winning_player", &mali_ba::Mali_BaState::GetWinningPlayer)
        .def("get_game_end_triggering_player", &mali_ba::Mali_BaState::GetGameEndTriggeringPlayer)
        .def("get_score_breakdown_string", &mali_ba::Mali_BaState::GetScoreBreakdownString)
        .def("current_phase", &mali_ba::Mali_BaState::CurrentPhase)
        .def("is_near_win", &mali_ba::Mali_BaState::IsNearWin,
             py::arg("rare_region_threshold") = 4)
        .def("seed_rng", [](mali_ba::Mali_BaState& s, uint32_t seed) {
            s.GetRNG().seed(seed);
        }, "Re-seed the state's internal RNG for per-episode randomness.");

    // Game class - use py::classh to match base Game registration in pyspiel.cc
    py::classh<mali_ba::Mali_BaGame, open_spiel::Game> game_class_binder(m, "Mali_BaGame");
    game_class_binder
        .def("deserialize_state", &mali_ba::Mali_BaGame::DeserializeState)
        .def("get_grid_radius", &mali_ba::Mali_BaGame::GetGridRadius)
        .def("get_valid_hexes", [](const mali_ba::Mali_BaGame& game) {
            const auto& hex_set = game.GetValidHexes();
            return std::vector<mali_ba::HexCoord>(hex_set.begin(), hex_set.end());
        })
        .def("get_cities", &mali_ba::Mali_BaGame::GetCities, py::return_value_policy::reference_internal)
        .def("get_training_parameters", &mali_ba::Mali_BaGame::GetTrainingParameters, py::return_value_policy::reference_internal)
        .def("new_initial_state",
            static_cast<std::unique_ptr<open_spiel::State> (open_spiel::mali_ba::Mali_BaGame::*)() const>(
                &mali_ba::Mali_BaGame::NewInitialState
            ))
        .def("new_initial_state",
             static_cast<std::unique_ptr<open_spiel::State> (open_spiel::mali_ba::Mali_BaGame::*)(const std::string&) const>(
                 &mali_ba::Mali_BaGame::NewInitialState
             ));

    // Downcast helpers: convert base Game/State pointers to Mali_Ba-specific types
    mali_ba.def("downcast_game", [](std::shared_ptr<const open_spiel::Game> game)
            -> std::shared_ptr<const mali_ba::Mali_BaGame> {
        return std::dynamic_pointer_cast<const mali_ba::Mali_BaGame>(game);
    });
    mali_ba.def("downcast_state", [](std::shared_ptr<open_spiel::State> state)
            -> std::shared_ptr<mali_ba::Mali_BaState> {
        return std::dynamic_pointer_cast<mali_ba::Mali_BaState>(state);
    });

    // Logging
    py::enum_<mali_ba::LogLevel>(mali_ba, "LogLevel")
        .value("DEBUG",   mali_ba::LogLevel::kDebug)
        .value("INFO",    mali_ba::LogLevel::kInfo)
        .value("WARN",    mali_ba::LogLevel::kWarning)
        .value("ERROR",   mali_ba::LogLevel::kError)
        .export_values();

    mali_ba.def("log", &mali_ba::LogFromPython);
    mali_ba.def("set_log_level", &mali_ba::SetLogLevel);

    // Utility functions
    mali_ba.def("player_color_to_string", &mali_ba::PlayerColorToString);
    mali_ba.def("string_to_player_color", &mali_ba::StringToPlayerColor);
    mali_ba.def("meeple_color_to_string", &mali_ba::MeepleColorToString);
}

} // namespace open_spiel