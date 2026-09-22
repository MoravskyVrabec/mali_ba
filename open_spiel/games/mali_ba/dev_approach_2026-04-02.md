
# 🏗️ Mali-Ba Refactoring Attack Plan

This roadmap transitions Mali-Ba from a combinatorial, heavily Python-coupled prototype into a high-performance, AlphaZero-ready OpenSpiel engine with a cleanly decoupled GUI.

## ✅ Phase 1: C++ Core Game Logic & Action Space (Completed)
*The goal of this phase was to make the engine AI-friendly and modular.*
*   **Sequential FSM Action Space:** Deconstructed the massive ~50,000 compound action space into a finite state machine (Play $\rightarrow$ MancalaStep $\rightarrow$ Post $\rightarrow$ Route). Action space is now a highly efficient ~400 actions.
*   **Mid-Turn State Tracking:** Added lightweight, copyable C++ structures (`meeples_in_hand_`, `current_mancala_path_`) to handle mid-turn MCTS cloning perfectly.
*   **JSON Serialization Update:** Ensured mid-turn FSM state is perfectly serialized and deserialized.
*   **Clean Dependency Injection:** Transplanted the Mali-Ba engine into a pristine OpenSpiel fork using standard CMake targets, leaving the core framework untouched.

## 🚧 Phase 2: Python/C++ Interoperability (Current Phase)
*The goal of this phase is to eliminate the JSON bottleneck and make C++ the strict single source of truth.*
*   **Fix Python Environment Pathing:** Stop using `sys.path.append` hacks. Use standard Python package management (`setup.py` / `pip install -e .`) so the environment reliably finds the correct `pyspiel.so` binary.
*   **Eliminate JSON UI Parsing:** Expand `games_mali_ba.cc` (Pybind11) so Python can query the board directly (e.g., `state.get_meeples_at(hex)`). Delete `GameStateCache` and `parse_and_update_state_from_json` entirely.
*   **Centralize Rule Validation:** Remove all Python-side rule validation (e.g., `can_start_mancala_at`). The UI must strictly query `state.legal_actions()` to determine what is clickable, preventing UI/Engine desyncs.

## ⏳ Phase 3: Python GUI & Visualizer Refactoring
*The goal of this phase is to organize the frontend into a clean MVC (Model-View-Controller) architecture.*
*   **Decompose `visualizer.py`:** Split the massive Visualizer class into three distinct components: 
    *   *Input Handler* (Mouse clicks / Event loop).
    *   *Engine Controller* (Sending actions to C++).
    *   *Renderer* (Drawing the Pygame screen).
*   **Modularize UI Widgets:** Break up `gui_other.py` and `visualizer_other.py`. Move `Sidebar`, `ControlPanel`, `InteractiveObject`, and `DialogBox` into their own dedicated files inside a `ui/widgets/` directory.

## ⏳ Phase 4: AI Training Loop Optimization
*The goal of this phase is to scale the AlphaZero training loop for maximum GPU utilization.*
*   **Batched GPU Inference:** Remove the Keras models from the individual CPU `actor_processes`. Create a centralized GPU inference process/thread. Actors will push observations to a queue, and the GPU will process them in batches of 64/128, returning predictions instantly.
*   **Python Anti-Pattern Cleanup:** Replace all bare `except:` clauses in the queues with specific `except queue.Empty:` to stop hiding real crashes.
*   **Dead Code Eradication:** Aggressively delete the massive blocks of commented-out legacy architecture in `main.py` and `train_mali_ba.py` to make the training loop readable.

---

### 🔍 Quick Fix for your `AttributeError`

The error `module 'pyspiel' has no attribute 'mali_ba'` means Python is successfully importing `pyspiel`, but it is **importing the wrong one** (likely a pre-compiled pip version in your conda environment, or an old build). 

You can prove this by running this exact command in your terminal:
```bash
python -c "import pyspiel; print(pyspiel.__file__)"
```
If it prints something like `/home/robp/miniconda3/envs/mali_ba/lib/python3.11/site-packages/pyspiel...`, it is ignoring your build folder completely.

**To fix this permanently (since you have a `setup.py`!):**
1. Uninstall any rogue pip versions of open_spiel: 
   ```bash
   pip uninstall open_spiel
   ```
2. Force your terminal to prioritize your new build directory by exporting the `PYTHONPATH` before you run the script:
   ```bash
   export PYTHONPATH=/media/robp/UD/Projects/open_spiel/build/python:$PYTHONPATH
   python mali_ba/main.py --mode cpp_sync_gui
   ```

Let me know what that `print(pyspiel.__file__)` command outputs, and we will get this UI hooked up to your shiny new C++ engine!
