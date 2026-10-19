"""Playing Mali-Ba: a game session that any front end can drive (local GUI, web, phone).

The C++ engine (pyspiel) holds every rule. This package adds, in pure Python:
  session.py   GameSession -- one game: seats (humans or bots), turn flow, the legal
               moves at each step described for display, applying moves, the result
  describe.py  turns an engine action into something a front end can draw
  bots.py      heuristic and network (MCTS + trained network) players

No front end needs game rules: it shows what GameSession.legal_moves() returns and
sends one choice at a time. See docs/design/ENGINE_UI_SEPARATION.md.
"""
