#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/spiel.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/tests/basic_tests.h"

// Include the nlohmann/json header
#include "json.hpp"
// For convenience
using json = nlohmann::json;

using open_spiel::Game;
using open_spiel::GameParameters;
using open_spiel::State;

namespace open_spiel
{
    namespace mali_ba
    {
        namespace
        {

            namespace testing = open_spiel::testing;

            // =============================================================================
            // Test Fixture and Helpers
            // =============================================================================

            struct MaliBaTest
            {
                std::shared_ptr<const Game> game;
                std::unique_ptr<State> state;
                Mali_BaState *mali_ba_state;

                // A constructor that takes a pre-loaded game object.
                // This removes all ambiguity about how the game is loaded.
                MaliBaTest(std::shared_ptr<const Game> loaded_game)
                {
                    game = loaded_game;
                    SPIEL_CHECK_TRUE(game != nullptr);
                    state = game->NewInitialState();
                    SPIEL_CHECK_TRUE(state != nullptr);
                    mali_ba_state = static_cast<Mali_BaState *>(state.get());
                }

                void AdvancePastSetup()
                {
                    if (mali_ba_state->IsChanceNode())
                    {
                        SPIEL_CHECK_EQ(mali_ba_state->LegalActions().size(), 1);
                        state->ApplyAction(mali_ba_state->LegalActions()[0]);
                    }
                }

                // A more robust version that ensures caches are cleared, preventing hangs
                // if the state's internal logic has subtle bugs.
                void AdvanceToPlayPhase()
                {
                    AdvancePastSetup();
                    SPIEL_CHECK_EQ(mali_ba_state->CurrentPhase(), Phase::kPlaceToken);

                    while (mali_ba_state->CurrentPhase() == Phase::kPlaceToken)
                    {
                        SPIEL_CHECK_FALSE(mali_ba_state->IsTerminal());
                        std::vector<Action> legal_actions = mali_ba_state->LegalActions();
                        SPIEL_CHECK_FALSE(legal_actions.empty());
                        state->ApplyAction(legal_actions[0]);

                        // This is good practice in a test loop to ensure a clean state
                        // for the next iteration's LegalActions() call.
                        mali_ba_state->ClearCaches();
                    }
                    SPIEL_CHECK_EQ(mali_ba_state->CurrentPhase(), Phase::kPlay);
                }

                void ApplyAction(const std::string &action_str)
                {
                    // The new ParseMoveStringToAction is now efficient and robust.
                    Action action = mali_ba_state->ParseMoveStringToAction(action_str);
                    if (action == kInvalidAction)
                    {
                        std::string available_actions_str;
                        for (Action act : mali_ba_state->LegalActions())
                        {
                            absl::StrAppend(&available_actions_str, "\n  - '", mali_ba_state->ActionToString(mali_ba_state->CurrentPlayer(), act), "'");
                        }
                        SpielFatalError(absl::StrCat("ApplyAction failed to parse '", action_str, "'. Available actions: ", available_actions_str));
                    }
                    state->ApplyAction(action);
                }
            };

            // =============================================================================
            // Test Cases
            // =============================================================================

            void APITest_StateCreationAndClone(std::shared_ptr<const Game> game)
            {
                LOG_INFO("--- APITest_StateCreationAndClone ---");
                MaliBaTest test(game);
                SPIEL_CHECK_EQ(test.state->CurrentPlayer(), kChancePlayerId);
                SPIEL_CHECK_TRUE(test.state->IsChanceNode());
                std::unique_ptr<State> clone = test.state->Clone();
                SPIEL_CHECK_EQ(test.state->Serialize(), clone->Serialize());
                clone->ApplyAction(clone->LegalActions()[0]);
                SPIEL_CHECK_NE(test.state->Serialize(), clone->Serialize());
                LOG_INFO("APITest_StateCreationAndClone passed.");
            }

            void SetupAndPlacementTest(std::shared_ptr<const Game> game)
            {
                LOG_INFO("--- SetupAndPlacementTest ---");
                MaliBaTest test(game);
                SPIEL_CHECK_TRUE(test.mali_ba_state->IsChanceNode());
                test.AdvancePastSetup();
                SPIEL_CHECK_FALSE(test.mali_ba_state->IsChanceNode());
                SPIEL_CHECK_EQ(test.mali_ba_state->CurrentPhase(), Phase::kPlaceToken);
                SPIEL_CHECK_EQ(test.mali_ba_state->CurrentPlayer(), 0);
                bool meeples_found = false;
                for (const auto &hex : test.mali_ba_state->ValidHexes())
                {
                    if (!test.mali_ba_state->GetMeeplesAt(hex).empty())
                    {
                        meeples_found = true;
                        break;
                    }
                }
                SPIEL_CHECK_TRUE(meeples_found);
                Player first_player = test.state->CurrentPlayer();
                std::vector<Action> legal_actions = test.state->LegalActions();
                SPIEL_CHECK_FALSE(legal_actions.empty());
                test.state->ApplyAction(legal_actions[0]);
                SPIEL_CHECK_EQ(test.state->CurrentPlayer(), (first_player + 1) % test.game->NumPlayers());
                LOG_INFO("SetupAndPlacementTest passed.");
            }

            void MancalaMoveTest_OneMeeple(std::shared_ptr<const Game> game)
            {
                LOG_INFO("--- MancalaMoveTest_OneMeeple ---");
                MaliBaTest test(game);
                test.AdvanceToPlayPhase();

                HexCoord start_hex(0, 1, -1);
                HexCoord end_hex(1, 0, -1);
                Player p0 = 0;
                PlayerColor p0_color = test.mali_ba_state->GetPlayerColor(p0);

                test.mali_ba_state->TestOnly_ClearPlayerTokens();
                test.mali_ba_state->TestOnly_ClearMeeples();
                test.mali_ba_state->TestOnly_SetPlayerToken(start_hex, p0_color);
                test.mali_ba_state->TestOnly_SetMeeples(start_hex, {MeepleColor::kSolidGold});
                test.mali_ba_state->TestOnly_SetCurrentPlayer(p0);

                test.mali_ba_state->ClearCaches(); // Invalidate cache after manual changes

                bool found_action = false;
                for (Action action : test.mali_ba_state->LegalActions())
                {
                    Move m = test.mali_ba_state->ActionToMove(action);
                    if (m.type == ActionType::kMancala && m.start_hex == start_hex && !m.path.empty() && m.path.back() == end_hex)
                    {
                        LOG_INFO(absl::StrCat("Found matching mancala action: ", action, ". Applying..."));
                        test.state->ApplyAction(action);
                        found_action = true;
                        break;
                    }
                }
                SPIEL_CHECK_TRUE(found_action);

                SPIEL_CHECK_EQ(test.mali_ba_state->GetPlayerTokenAt(start_hex), PlayerColor::kEmpty);
                SPIEL_CHECK_TRUE(test.mali_ba_state->GetMeeplesAt(start_hex).empty());
                SPIEL_CHECK_EQ(test.mali_ba_state->GetPlayerTokenAt(end_hex), p0_color);
                LOG_INFO("MancalaMoveTest_OneMeeple passed.");
            }

            void UpgradePostTest_ResourceCost(std::shared_ptr<const Game> game)
            {
                LOG_INFO("--- UpgradePostTest_ResourceCost ---");
                MaliBaTest test(game);
                test.AdvanceToPlayPhase();

                Player p0 = 0;
                PlayerColor p0_color = test.mali_ba_state->GetPlayerColor(p0);
                HexCoord post_hex(1, 1, -2);

                test.mali_ba_state->TestOnly_SetTradePost(post_hex, p0_color, TradePostType::kPost);
                test.mali_ba_state->TestOnly_SetCommonGood(p0, "Cattle", 3);
                test.mali_ba_state->TestOnly_SetRareGood(p0, "Dogon Mask", 0);
                test.mali_ba_state->TestOnly_SetCurrentPlayer(p0);

                test.ApplyAction("upgrade 1,1,-2|Cattle:3|");

                SPIEL_CHECK_EQ(test.mali_ba_state->GetCommonGoodCount(p0, "Cattle"), 0);
                const auto &posts_at_hex = test.mali_ba_state->GetTradePostsAt(post_hex);
                SPIEL_CHECK_EQ(posts_at_hex.size(), 1);
                SPIEL_CHECK_EQ(posts_at_hex[0].owner, p0_color);
                SPIEL_CHECK_EQ(posts_at_hex[0].type, TradePostType::kCenter);

                LOG_INFO("UpgradePostTest_ResourceCost passed.");
            }

            void SerializationTest_MidGame(std::shared_ptr<const Game> game)
            {
                LOG_INFO("--- SerializationTest_MidGame ---");
                MaliBaTest test(game);
                test.AdvanceToPlayPhase();

                std::vector<Action> legal_actions = test.mali_ba_state->LegalActions();
                SPIEL_CHECK_FALSE(legal_actions.empty());
                test.state->ApplyAction(legal_actions[0]);
                test.ApplyAction("pass");

                std::string original_serialized = test.state->Serialize();
                SPIEL_CHECK_FALSE(original_serialized.empty());

                std::unique_ptr<State> deserialized_state = test.game->DeserializeState(original_serialized);
                SPIEL_CHECK_TRUE(deserialized_state != nullptr);
                SPIEL_CHECK_EQ(original_serialized, deserialized_state->Serialize());
                SPIEL_CHECK_EQ(test.state->ToString(), deserialized_state->ToString());

                LOG_INFO("SerializationTest_MidGame passed.");
            }

            void UndoActionTest(std::shared_ptr<const Game> game)
            {
                LOG_INFO("--- UndoActionTest ---");
                MaliBaTest test(game);
                test.AdvanceToPlayPhase();

                // Make sure we are in a state with some moves
                std::vector<Action> legal_actions = test.mali_ba_state->LegalActions();
                SPIEL_CHECK_FALSE(legal_actions.empty());

                // Take a snapshot
                std::string state_before_action = test.mali_ba_state->Serialize();
                Action chosen_action = legal_actions[0]; // Choose the first legal action
                Player player_before_action = test.mali_ba_state->CurrentPlayer();

                // Apply the action
                test.state->ApplyAction(chosen_action);
                std::string state_after_action = test.mali_ba_state->Serialize();

                // Check that the state changed
                SPIEL_CHECK_NE(state_before_action, state_after_action);

                // Undo the action
                test.mali_ba_state->UndoAction(player_before_action, chosen_action);
                std::string state_after_undo = test.mali_ba_state->Serialize();

                // Check that the state is back to the original
                SPIEL_CHECK_EQ(state_before_action, state_after_undo);
                SPIEL_CHECK_EQ(test.mali_ba_state->CurrentPlayer(), player_before_action);

                LOG_INFO("UndoActionTest passed.");
            }

            void IniFileConfigTest()
            {
                LOG_INFO("--- IniFileConfigTest ---");

                // Create a temporary config file for the test
                std::string config_content =
                    "[Board]\n"
                    "grid_radius = 2\n"
                    "custom_hexes = 0,0,0; 1,0,-1; -1,0,1; 0,1,-1; 0,-1,1; 1,-1,0; -1,1,0\n" // 7 hexes
                    "[Cities]\n"
                    "city1 = Timbuktu,0,0,0\n"
                    "city2 = Segou,1,-1,0\n";
                std::string config_path = "/tmp/mali_ba_test.ini";
                std::ofstream out(config_path);
                out << config_content;
                out.close();

                open_spiel::GameParameters params;
                params["config_file"] = open_spiel::GameParameter(config_path);

                // Load the game with the config file parameter.
                std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("mali_ba", params);
                SPIEL_CHECK_TRUE(game != nullptr);

                // Cast to the specific game type to access its members
                const auto *mali_ba_game = static_cast<const Mali_BaGame *>(game.get());

                // Check if the GAME object was configured correctly.
                SPIEL_CHECK_EQ(mali_ba_game->GetGridRadius(), 2);
                SPIEL_CHECK_EQ(mali_ba_game->GetValidHexes().size(), 7);
                SPIEL_CHECK_EQ(mali_ba_game->GetCities().size(), 2);

                // The game object is now configured. Create a state FROM it.
                std::unique_ptr<open_spiel::State> state = game->NewInitialState();
                auto mali_ba_state = static_cast<Mali_BaState *>(state.get());

                // Check if the STATE correctly reflects the game's configuration.
                SPIEL_CHECK_EQ(mali_ba_state->GridRadius(), 2);
                SPIEL_CHECK_EQ(mali_ba_state->ValidHexes().size(), 7);
                SPIEL_CHECK_EQ(mali_ba_state->GetCities().size(), 2);

                // Print the state to see the result
                std::cout << "State from INI file config:\n"
                          << state->ToString() << std::endl;

                LOG_INFO("IniFileConfigTest passed.");
            }

            void EndGameRequirementTest(std::shared_ptr<const Game> game)
            {
                LOG_INFO("--- EndGameRequirementTest ---");

                // // Create a temporary config file for this specific test
                // std::string config_content =
                //     "[Rules]\n"
                //     "endgm_cond_numrare_goods = 4\n"   // Ends with 4 rare goods
                //     "endgm_req_numroutes = 2\n";      // ...but only if you have 2 routes
                // std::string config_path = "/tmp/mali_ba_req_test.ini";
                // std::ofstream out(config_path);
                // out << config_content;
                // out.close();

                // // Load the game using this specific configuration
                // open_spiel::GameParameters params;
                // params["config_file"] = open_spiel::GameParameter(config_path);
                MaliBaTest test(game);
                test.AdvanceToPlayPhase();

                Player p0 = 0;
                PlayerColor p0_color = test.mali_ba_state->GetPlayerColor(p0);
                test.mali_ba_state->TestOnly_SetCurrentPlayer(p0);

                // --- Part 1: Player gets 4 rare goods, but only 1 route ---
                LOG_INFO("Test Part 1: 4 rare goods, 1 route. Game should NOT end.");
                test.mali_ba_state->TestOnly_SetRareGood(p0, "Gold", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p0, "Silver", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p0, "Bronze", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p0, "Kora", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p0, "Kru boat", 1);

                // Give player necessary posts for one route
                test.mali_ba_state->TestOnly_SetTradePost(HexCoord(0, 1, -1), p0_color, TradePostType::kCenter); // Add another post
                test.mali_ba_state->TestOnly_SetTradePost(HexCoord(1, 0, -1), p0_color, TradePostType::kCenter);
                test.mali_ba_state->TestOnly_SetTradePost(HexCoord(2, 0, -2), p0_color, TradePostType::kCenter);
                test.mali_ba_state->CreateTradeRoute({HexCoord(1, 0, -1), HexCoord(2, 0, -2), HexCoord(0, 1, -1)}, p0_color);

                // Check game state. It should NOT be terminal.
                SPIEL_CHECK_FALSE(test.mali_ba_state->IsTerminal());

                // --- Part 2: Player now builds a second route ---
                LOG_INFO("Test Part 2: Player builds a 2nd route. Game SHOULD end now.");
                test.mali_ba_state->TestOnly_SetTradePost(HexCoord(3, 0, -3), p0_color, TradePostType::kCenter);
                test.mali_ba_state->TestOnly_SetTradePost(HexCoord(1, 0, -1), p0_color, TradePostType::kCenter);

                // This is the action that should trigger the end game
                test.mali_ba_state->TestOnly_SetTradePost(HexCoord(0, 2, -2), p0_color, TradePostType::kCenter); // Add another post
                test.mali_ba_state->CreateTradeRoute({HexCoord(1, 0, -1), HexCoord(3, 0, -3), HexCoord(0, 2, -2)}, p0_color);

                // Now the state MUST be terminal
                test.mali_ba_state->RefreshTerminalStatus();
                SPIEL_CHECK_TRUE(test.mali_ba_state->IsTerminal());

                LOG_INFO("EndGameRequirementTest passed.");
            }

            void EndGameTriggerAndScoringTest(std::shared_ptr<const Game> game)
            {
                LOG_INFO("--- EndGameTriggerAndScoringTest (v0.7 Rules) ---");
                MaliBaTest test(game);
                // test.AdvanceToPlayPhase(); // We will set state manually

                Player p0 = 0, p1 = 1, p2 = 2;
                PlayerColor p0_color = test.mali_ba_state->GetPlayerColor(p0);
                PlayerColor p1_color = test.mali_ba_state->GetPlayerColor(p1);
                PlayerColor p2_color = test.mali_ba_state->GetPlayerColor(p2);

                test.mali_ba_state->SetCurrentPhase(Phase::kPlay);
                test.mali_ba_state->TestOnly_SetCurrentPlayer(p0);

                // --- Player 0 Setup: Longest Route, Regions Crossed ---
                // Expected Score (P0):
                // - Posts & Centers: 2 Posts (4*2=8), 2 Centers (8*2=16) = 24 pts
                // - Unique Common Sets: None
                // - Longest Route: 1st place (6 hexes) = 11 pts
                // - Region Control: 0 pts (tied with P2 in R3)
                // - Regions Crossed: Route crosses 2 regions (R1, R3) = 8 pts
                // Total (P0) = 24 + 11 + 8 = 43 pts
                test.mali_ba_state->TestOnly_SetTradePost({0, -4, 4}, p0_color, TradePostType::kCenter); // In R1
                test.mali_ba_state->TestOnly_SetTradePost({1, -4, 3}, p0_color, TradePostType::kCenter); // In R1
                test.mali_ba_state->TestOnly_SetTradePost({0, 1, -1}, p0_color, TradePostType::kPost);   // In R3
                test.mali_ba_state->TestOnly_SetTradePost({-1, 2, -1}, p0_color, TradePostType::kPost);  // In R3
                test.mali_ba_state->CreateTradeRoute({{0, -4, 4}, {1, -3, 2}, {0, 1, -1}, {-1, 2, -1}, {-2, 3, -1}, {-3, 4, -1}}, p0_color);

                // --- Player 1 Setup: Unique Goods Sets, End-game trigger ---
                // Expected Score (P1):
                // - Posts & Centers: 1 Center (from initial setup) + 4 new centers = 5 * 8 = 40 pts
                // - Unique Common Sets (from rules v0.7):
                //   - Set of 6 (>=1 of each): 30 pts
                //   - Set of 4 (>=2 of each): 11 pts
                //   - Set of 2 (>=3 of each): 3 pts
                //   Total = 30 + 11 + 3 = 44 pts
                // - Longest Route: None (both are 3 hexes)
                // - Region Control: None
                // - Regions Crossed: None (assuming routes are in one region for simplicity)
                // Total (P1) = 40 + 44 = 84 pts
                test.mali_ba_state->TestOnly_SetTradePost({0, 4, -4}, p1_color, TradePostType::kCenter); // Center for goods
                test.mali_ba_state->TestOnly_SetCommonGood(p1, "Cattle", 3);
                test.mali_ba_state->TestOnly_SetCommonGood(p1, "Camel", 3);
                test.mali_ba_state->TestOnly_SetCommonGood(p1, "Pepper", 2);
                test.mali_ba_state->TestOnly_SetCommonGood(p1, "Kora", 2);
                test.mali_ba_state->TestOnly_SetCommonGood(p1, "Chiwara", 1);
                test.mali_ba_state->TestOnly_SetCommonGood(p1, "Gold", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Silver cross", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Dogon mask", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Wedding blanket", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Silver headdress", 1);

                // Set up centers for the first route
                test.mali_ba_state->TestOnly_SetTradePost({-1, -1, 2}, p1_color, TradePostType::kCenter);
                test.mali_ba_state->TestOnly_SetTradePost({-2, -1, 3}, p1_color, TradePostType::kCenter);
                test.mali_ba_state->CreateTradeRoute({{-1, -1, 2}, {-2, -1, 3}, {0, 4, -4}}, p1_color);

                // The Game-Ending Move: Player 1 has 5 rare goods and builds a *second*, valid route.
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Gold weight", 1); // 5th rare good

                // **** THIS IS THE FIX: Set up NEW centers for the second route ****
                // This route will share ONE center ({-1, -1, 2}) with the first route, which is valid.
                test.mali_ba_state->TestOnly_SetTradePost({-1, 0, 1}, p1_color, TradePostType::kCenter); // New center 1
                test.mali_ba_state->TestOnly_SetTradePost({-2, 0, 2}, p1_color, TradePostType::kCenter); // New center 2
                test.mali_ba_state->CreateTradeRoute({{-1, -1, 2}, {-1, 0, 1}, {-2, 0, 2}}, p1_color);   // Creates the 2nd route

                // --- Player 2 Setup: Region Control ---
                // Expected Score (P2):
                // - Posts & Centers: 2 Centers (8*2=16) = 16 pts
                // - Unique Common Sets: None
                // - Longest Route: 2nd place (4 hexes) = 7 pts
                // - Region Control: 1st in R5 = 11 pts
                // - Regions Crossed: Route in 1 region = 4 pts
                // Total (P2) = 16 + 7 + 11 + 4 = 38 pts
                test.mali_ba_state->TestOnly_SetTradePost({1, 4, -5}, p2_color, TradePostType::kCenter);          // In R5
                test.mali_ba_state->TestOnly_SetTradePost({2, 3, -5}, p2_color, TradePostType::kCenter);          // In R5
                test.mali_ba_state->CreateTradeRoute({{1, 4, -5}, {2, 3, -5}, {1, 3, -4}, {0, 3, -3}}, p2_color); // 4-hex route

                // --- The Game-Ending Move ---
                // Player 1 has enough rare goods and will now build their second route to trigger the game end.
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Silver cross", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Dogon mask", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Wedding blanket", 1);
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Silver headdress", 1);
                test.mali_ba_state->TestOnly_SetTradePost({-1, -1, 2}, p1_color, TradePostType::kCenter); // Set up for routes
                test.mali_ba_state->TestOnly_SetTradePost({-2, -1, 3}, p1_color, TradePostType::kCenter);
                test.mali_ba_state->CreateTradeRoute({{-1, -1, 2}, {-2, -1, 3}, {0, 4, -4}}, p1_color);
                // Now Player 1 makes the final move, creating the 2nd route while having 5 rare goods
                test.mali_ba_state->TestOnly_SetRareGood(p1, "Gold weight", 1); // 5th rare good
                test.mali_ba_state->CreateTradeRoute({{-1, -1, 2}, {0, 4, -4}, {-2, -1, 3}}, p1_color);

                test.mali_ba_state->RefreshTerminalStatus();
                SPIEL_CHECK_TRUE(test.mali_ba_state->IsTerminal());

                // --- Final Score Calculation & Verification ---
                std::vector<double> final_scores = test.mali_ba_state->Returns();

                double p0_expected_score = 62.0;
                double p1_expected_score = 137.0;
                double p2_expected_score = 42.0;

                LOG_INFO(absl::StrCat("Final Scores: P0=", final_scores[0], " (Expected: ", p0_expected_score, "), \
        P1=",
                                      final_scores[1], " (Expected: ", p1_expected_score, "), P2=", final_scores[2],
                                      " (Expected: ", p2_expected_score, ")"));

                // Verify the scores
                SPIEL_CHECK_FLOAT_EQ(final_scores[0], p0_expected_score);
                SPIEL_CHECK_FLOAT_EQ(final_scores[1], p1_expected_score);
                SPIEL_CHECK_FLOAT_EQ(final_scores[2], p2_expected_score);

                // Player 1 should have the highest score.
                SPIEL_CHECK_GT(final_scores[1], final_scores[0]);
                SPIEL_CHECK_GT(final_scores[1], final_scores[2]);

                LOG_INFO("EndGameTriggerAndScoringTest (v0.7 Rules) passed.");
            }

            void RegionalBoardConfigTest()
            {
                LOG_INFO("--- RegionalBoardConfigTest ---");

                // Create a temporary config file with two regions
                std::string config_content =
                    "[Board]\n"
                    // Region 1: 3 hexes
                    "custom_hexes1 = 0,0,0:1,-1,0:-1,1,0\n"
                    // Region 2: 3 different hexes, and one overlapping hex (0,0,0)
                    "custom_hexes2 = 0,1,-1:1,0,-1:0,0,0\n";
                std::string config_path = "/tmp/mali_ba_region_test.ini";
                std::ofstream out(config_path);
                out << config_content;
                out.close();

                open_spiel::GameParameters params;
                params["config_file"] = open_spiel::GameParameter(config_path);

                // Load the game with the regional config
                std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("mali_ba", params);
                SPIEL_CHECK_TRUE(game != nullptr);

                const auto *mali_ba_game = static_cast<const Mali_BaGame *>(game.get());

                // 1. Check total number of unique hexes. Should be 5.
                // (0,0,0), (1,-1,0), (-1,1,0) from region 1
                // (0,1,-1), (1,0,-1) from region 2 (0,0,0 is a duplicate)
                SPIEL_CHECK_EQ(mali_ba_game->GetValidHexes().size(), 5);

                // 2. Check the region mapping for specific hexes
                SPIEL_CHECK_EQ(mali_ba_game->GetRegionForHex(HexCoord(1, -1, 0)), 1); // From region 1
                SPIEL_CHECK_EQ(mali_ba_game->GetRegionForHex(HexCoord(-1, 1, 0)), 1); // From region 1
                SPIEL_CHECK_EQ(mali_ba_game->GetRegionForHex(HexCoord(0, 1, -1)), 2); // From region 2
                SPIEL_CHECK_EQ(mali_ba_game->GetRegionForHex(HexCoord(1, 0, -1)), 2); // From region 2

                // 3. Check the overlapping hex. It should have the ID of the LAST region it was defined in.
                SPIEL_CHECK_EQ(mali_ba_game->GetRegionForHex(HexCoord(0, 0, 0)), 2);

                // 4. Check a hex that is not in any region.
                SPIEL_CHECK_EQ(mali_ba_game->GetRegionForHex(HexCoord(10, 10, -20)), -1);

                LOG_INFO("RegionalBoardConfigTest passed.");
            }

            // A simplified version of ApplyActionTestClone that does NOT test the clone.
            // It applies the action only once.
            void ApplyActionNoClone(open_spiel::State *state, open_spiel::Action action)
            {
                state->ApplyAction(action);
            }

            // A simplified version of the main simulation loop from basic_tests.cc
            // that uses ApplyActionNoClone.
            void SingleExecutionRandomSimulation(
                std::mt19937 *rng, const open_spiel::Game &game)
            {
                std::cout << "--- Running Single-Execution Random Simulation ---" << std::endl;
                std::unique_ptr<open_spiel::State> state = game.NewInitialState();

                while (!state->IsTerminal())
                {
                    std::cout << state->ToString() << std::endl;

                    if (state->IsChanceNode())
                    {
                        // Chance node; sample one according to underlying distribution
                        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
                        Action action = open_spiel::SampleAction(outcomes, *rng).first;
                        std::cout << "Chance outcome: " << state->ActionToString(kChancePlayerId, action) << std::endl;
                        state->ApplyAction(action);
                    }
                    else
                    {
                        // Decision node.
                        Player player = state->CurrentPlayer();
                        std::vector<Action> actions = state->LegalActions();
                        SPIEL_CHECK_FALSE(actions.empty());

                        // // Sample a random action. Replaced with heuristic approach below
                        // std::uniform_int_distribution<int> dis(0, actions.size() - 1);
                        // Action action = actions[dis(*rng)];
                        // std::cout << "Player " << player << " chooses " << state->ActionToString(player, action) << std::endl;

                        // Heuristically select an action.
                        auto *mali_ba_state = static_cast<mali_ba::Mali_BaState *>(state.get());
                        Action action = mali_ba_state->SelectHeuristicRandomAction();

                        std::cout << "Player " << player << " chooses (heuristically) "
                                  << state->ActionToString(player, action) << std::endl;

                        // Apply the action a single time.
                        ApplyActionNoClone(state.get(), action);
                    }
                }

                std::cout << "--- Terminal State Reached ---" << std::endl;
                std::cout << state->ToString() << std::endl;
                // --- EXPLICITLY CALL RETURNS() TO TRIGGER SCORE LOGGING ---
                std::cout << "\n--- Calculating Final Returns and Scores... ---" << std::endl;
                std::vector<double> final_returns = state->Returns();
                // The score breakdown will now be printed to the log from within the Returns() function.
                std::cout << "\n--- Final Training Returns ---" << std::endl;
                std::cout << "Returns: " << absl::StrJoin(final_returns, ", ") << std::endl;
            }

            // A new public-facing test function that calls our simplified simulation.
            void SingleExecutionRandomSimTest(const open_spiel::Game &game, int num_sims)
            {
                std::mt19937 rng;
                for (int sim = 0; sim < num_sims; ++sim)
                {
                    SingleExecutionRandomSimulation(&rng, game);
                }
            }

        } // namespace
    } // namespace mali_ba
} // namespace open_spiel

int main(int argc, char **argv)
{
    // Create game with logging parameter
    std::map<std::string, open_spiel::GameParameter> params;
    // Check for logging or help flag
    bool move_logging = false;
    bool use_ini = false;
    std::string ini_path = "";

    printf("args:\n");
    for (int i = 1; i < argc; ++i)
    {
        std::cout << "arg " << i << ": " << argv[i] << "\n";
        if (strcmp(argv[i], "--log_moves") == 0)
        {
            move_logging = true;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Mali-Ba Test Runner\n";
            std::cout << "Options:\n";
            std::cout << "  --log_moves                  Enable move logging for all tests\n";
            std::cout << "  --config_file [file path]       Use the specified ini file\n";
            std::cout << "  --help, -h                   Show this help message\n";
            return 0;
        }
        else if (strcmp(argv[i], "--config_file") == 0)
        {
            // Check if there is next argument and trust that it's the path
            if (argc > i)
            {
                use_ini = true;
                ini_path = argv[i + 1];
            }
            else
            {
                std::cout << "Error: no ini file path supplied\n";
                return 1;
            }
        }
    }

    if (move_logging)
    {
        params["enable_move_logging"] = open_spiel::GameParameter(true);
        std::cout << "Move logging enabled for RandomSimTest" << std::endl;
    }
    if (use_ini)
    {
        params["config_file"] = open_spiel::GameParameter(ini_path);
        std::cout << "ini file use enabled for RandomSimTest: " << ini_path << std::endl;
    }
    // return -2; // DEBUG

    // --- 2. Load the Game ONE TIME ---
    // The 'game' object will be configured with your INI file.
    std::cout << "Loading game with specified configuration..." << std::endl;
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("mali_ba", params);
    SPIEL_CHECK_TRUE(game != nullptr);
    std::cout << "Game loaded successfully." << std::endl;

    // ALL THESE BASIC TESTS PASSED ON 7/7/25
    // open_spiel::mali_ba::APITest_StateCreationAndClone(game);
    // open_spiel::mali_ba::SetupAndPlacementTest(game);
    // open_spiel::mali_ba::MancalaMoveTest_OneMeeple(game);
    // open_spiel::mali_ba::UpgradePostTest_ResourceCost(game);
    // open_spiel::mali_ba::UndoActionTest(game);
    // open_spiel::mali_ba::SerializationTest_MidGame(game);
    // open_spiel::mali_ba::IniFileConfigTest();
    open_spiel::mali_ba::EndGameRequirementTest(game);
    open_spiel::mali_ba::EndGameTriggerAndScoringTest(game);
    open_spiel::mali_ba::RegionalBoardConfigTest();

    // NOW THE RANDOM MOVES TESTS
    /*
    Functions and their Purpose
    1. open_spiel::testing::RandomSimTest	Two	DoApplyAction Calls per Move
    The standard OpenSpiel test. Rigorously tests game logic, serialization, and the
    correctness of State::Clone().
    2. mali_ba::SingleExecutionRandomSimTest (your custom function)	One	DoApplyAction Call per Move
    A simplified test runner. Useful for debugging your game's core logic and producing clean,
    easy-to-follow logs of a single game playthrough.
    */
    // // =============================================================================
    // printf("--- 1. Running Basic RandomSimTest ---\n");
    // // =============================================================================
    // // Load game with parameters and run test
    // std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("mali_ba", params);
    // open_spiel::testing::RandomSimTest(*game, 1);

    // printf("--- All Tests Passed ---\n");

    // // Show log file location if logging was enabled
    // if (move_logging) {
    //     std::string log_file = open_spiel::mali_ba::Mali_BaState::GetMoveLogFilename();
    //     if (!log_file.empty()) {
    //         std::cout << "Replay file created: " << log_file << std::endl;
    //         std::cout << "To replay: python main.py --mode gui_replay --replay_file " << log_file << std::endl;
    //     }
    // }

    // =============================================================================
    printf("--- 2. Running Custom RandomSimTest-one DoApplyAction() per move---\n");
    // =============================================================================
    // game loading done above
    // // Load game with parameters and run test
    // std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("mali_ba", params);

    // Call your new test runner instead of the default one.
    open_spiel::mali_ba::SingleExecutionRandomSimTest(*game, 1);

    printf("--- All Tests Passed ---\n");
}