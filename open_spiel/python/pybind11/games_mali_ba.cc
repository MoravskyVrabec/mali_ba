      
// --- START OF FILE games_mali_ba.cc ---

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
// No need for 'using open_spiel::mali_ba::...' here at the global scope
// if the function is inside the open_spiel namespace

// --- WRAP THE ENTIRE FUNCTION DEFINITION IN THE open_spiel NAMESPACE ---
namespace open_spiel {

// Now you can use the shorter names for Mali-Ba specific types if you wish,
// or continue using the fully qualified names.
// For clarity, let's use using declarations *inside* this function if needed,
// or just use the fully qualified names.

void init_pyspiel_games_mali_ba(::pybind11::module &m) {
    // Create a submodule
    py::module_ mali_ba = m.def_submodule("mali_ba");
    
    // Define constants
    m.attr("INVALID_ACTION") = py::int_(mali_ba::kInvalidAction); 
    m.attr("EMPTY_ACTION") = py::int_(mali_ba::kEmptyAction);
    m.attr("PASS_ACTION") = py::int_(mali_ba::kPassAction);
    m.attr("CHANCE_SETUP_ACTION") = py::int_(mali_ba::kChanceSetupAction); 
    m.attr("MAX_ACTION") = py::int_(mali_ba::kMaxAction); 
    
    // Enums
    py::enum_<open_spiel::mali_ba::Phase>(mali_ba, "Mali_BaPhase")
        .value("EMPTY", open_spiel::mali_ba::Phase::kEmpty)
        .value("SETUP", open_spiel::mali_ba::Phase::kSetup)
        .value("PLACE_TOKEN", open_spiel::mali_ba::Phase::kPlaceToken)
        .value("PLAY", open_spiel::mali_ba::Phase::kPlay)
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
        
    // ... Other enums like MeepleColor and TradePostType
    
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

    // Add the TradeRoute class
    py::class_<mali_ba::TradeRoute>(mali_ba, "TradeRoute")
        .def(py::init<>())
        .def_readonly("id", &mali_ba::TradeRoute::id)
        .def_readonly("owner", &mali_ba::TradeRoute::owner)
        .def_readonly("hexes", &mali_ba::TradeRoute::hexes)
        .def_readonly("goods", &mali_ba::TradeRoute::goods)
        .def_readonly("active", &mali_ba::TradeRoute::active);



        // State class - with pickle support
        py::class_<mali_ba::Mali_BaState, open_spiel::State, std::shared_ptr<mali_ba::Mali_BaState>> state_class_binder(m, "Mali_BaState");
        state_class_binder // Use the named variable to chain .def calls
            .def("play_random_move_and_serialize", &mali_ba::Mali_BaState::PlayRandomMoveAndSerialize)
            .def("get_player_common_goods", &mali_ba::Mali_BaState::GetPlayerCommonGoods, py::return_value_policy::reference_internal)
            .def("get_player_rare_goods", &mali_ba::Mali_BaState::GetPlayerRareGoods, py::return_value_policy::reference_internal)
            .def("parse_move_string_to_action", &mali_ba::Mali_BaState::ParseMoveStringToAction)
            .def("create_trade_route", &mali_ba::Mali_BaState::CreateTradeRoute)
            .def("update_trade_route", &mali_ba::Mali_BaState::UpdateTradeRoute)
            .def("delete_trade_route", &mali_ba::Mali_BaState::DeleteTradeRoute)
            .def("validate_trade_routes", &mali_ba::Mali_BaState::ValidateTradeRoutes)
            .def("apply_income_collection", &mali_ba::Mali_BaState::ApplyIncomeCollection)
            .def("serialize", &mali_ba::Mali_BaState::Serialize)
            // Pickle support for Mali_BaState
            .def(py::pickle(
                [](const mali_ba::Mali_BaState& state) -> std::string { // __getstate__
                    return SerializeGameAndState(*state.GetGame(), state);
                },
                [](const std::string& data) -> std::shared_ptr<mali_ba::Mali_BaState> { // __setstate__
                    std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>> game_and_state_pair = 
                        DeserializeGameAndState(data);
                    mali_ba::Mali_BaState* raw_state_ptr = 
                        dynamic_cast<mali_ba::Mali_BaState*>(game_and_state_pair.second.release());
                    if (!raw_state_ptr) {
                        throw std::runtime_error("DeserializeGameAndState did not return a Mali_BaState for State pickle.");
                    }
                    return std::shared_ptr<mali_ba::Mali_BaState>(raw_state_ptr);
                }
            )); // End of .def chain for state_class_binder (add semicolon if no more .def calls for it)

            // Game class - with pickle support
            py::class_<mali_ba::Mali_BaGame, open_spiel::Game, std::shared_ptr<mali_ba::Mali_BaGame>> game_class_binder(m, "Mali_BaGame");
            game_class_binder // Use the named variable to chain .def calls
                /*.def("get_type_copy", &mali_ba::Mali_BaGame::GetTypeCopy) */ // This method seems to be commented out
                .def("deserialize_state", &mali_ba::Mali_BaGame::DeserializeState) // This is a Mali_BaGame method
                .def("get_grid_radius", &mali_ba::Mali_BaGame::GetGridRadius)     // This is a Mali_BaGame method
                // Bind the no-argument NewInitialState
                .def("new_initial_state",
                    // Explicitly cast to the (std::unique_ptr<State> (YourClass::*)() const) version
                    static_cast<std::unique_ptr<open_spiel::State> (open_spiel::mali_ba::Mali_BaGame::*)() const>(
                        &mali_ba::Mali_BaGame::NewInitialState
                    ),
                    py::return_value_policy::move) // Apply policy if returning unique_ptr

                // OPTIONAL: Bind the NewInitialState(const std::string&) overload if you use/need it
                .def("new_initial_state",
                     // Explicitly cast to the (std::unique_ptr<State> (YourClass::*)(const std::string&) const) version
                     static_cast<std::unique_ptr<open_spiel::State> (open_spiel::mali_ba::Mali_BaGame::*)(const std::string&) const>(
                         &mali_ba::Mali_BaGame::NewInitialState
                     ),
                     py::return_value_policy::move) // Apply policy if returning unique_ptr
                
                // Pickle support for Mali_BaGame
                .def(py::pickle(
                    // For __getstate__, the argument should match the class being pickled
                    [](const mali_ba::Mali_BaGame& game) -> std::string { // __getstate__ for Mali_BaGame
                        // Or if you pass by shared_ptr:
                        // [](std::shared_ptr<mali_ba::Mali_BaGame> game) -> std::string {
                        return game.ToString(); // Assuming Game::ToString() is suitable for serializing Game
                    },
                    // For __setstate__, the return type should match the holder type
                    [](const std::string& data) -> std::shared_ptr<mali_ba::Mali_BaGame> { // __setstate__ for Mali_BaGame
                        return std::dynamic_pointer_cast<mali_ba::Mali_BaGame>(
                            std::const_pointer_cast<Game>(LoadGame(data)));
                    }
                )); // End of .def chain for game_class_binder (add semicolon if no more .def calls for it)

        
    // Utility functions
    mali_ba.def("player_color_to_string", &mali_ba::PlayerColorToString);
    mali_ba.def("string_to_player_color", &mali_ba::StringToPlayerColor);
    mali_ba.def("meeple_color_to_string", &mali_ba::MeepleColorToString);
}

} // namespace open_spiel
// --- END WRAPPER ---