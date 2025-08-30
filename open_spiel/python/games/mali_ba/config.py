# --- START OF FILE config.py ---

#import pygame
from enum import Enum

# --- Constants ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800
HEX_SIZE = 40  # Base size of hexagons
SIDEBAR_WIDTH = 300
CONTROLS_HEIGHT = 80 # Renamed from PANEL_HEIGHT for clarity
INFO_PANEL_HEIGHT = 70 # Height for individual player info panels
RESIZE_WAIT = 500
# Default number of players
DEFAULT_PLAYERS = 3
# Default grid radius
DEFAULT_GRID_RADIUS = 5
# Default number of tokens per player
DEFAULT_TOKENS_PER_PLAYER = 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (100, 100, 100)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (140, 140, 255)
DARK_BLUE = (80, 80, 255)
VIOLET = (200, 80, 255)
PINK = (255, 150, 200)
YELLOW = (255, 255, 80) # Used for highlighting
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
BRONZE = (205, 127, 50)
TAN = (210, 180, 140)

# --- Enums ---
class PlayerColor(Enum):
    EMPTY = -1
    RED = 0
    GREEN = 1
    BLUE = 2
    VIOLET = 3
    PINK = 4

    @classmethod
    def from_int(cls, value):
        for item in cls:
            if item.value == value:
                return item
        return cls.EMPTY # Default or error case

PLAYER_COLOR_DICT = {
    PlayerColor.RED: RED,
    PlayerColor.GREEN: GREEN,
    PlayerColor.BLUE: BLUE,
    PlayerColor.VIOLET: VIOLET,
    PlayerColor.PINK: PINK,
    PlayerColor.EMPTY: GRAY, # Added for completeness
}

class MeepleColor(Enum):
    EMPTY = -1
    SOLID_BLACK = 0
    CLEAR_BLACK = 1
    SOLID_SILVER = 2
    CLEAR_SILVER = 3
    CLEAR_WHITE = 4
    SOLID_GOLD = 5
    CLEAR_GOLD = 6
    SOLID_BRONZE = 7
    CLEAR_BRONZE = 8
    CLEAR_TAN = 9

    @classmethod
    def from_int(cls, value):
        for item in cls:
            if item.value == value:
                return item
        return cls.EMPTY

MEEPLE_COLOR_DICT = {
    MeepleColor.SOLID_BLACK: BLACK,
    MeepleColor.CLEAR_BLACK: (100, 100, 100),
    MeepleColor.SOLID_SILVER: SILVER,
    MeepleColor.CLEAR_SILVER: (220, 220, 220),
    MeepleColor.CLEAR_WHITE: WHITE,
    MeepleColor.SOLID_GOLD: GOLD,
    MeepleColor.CLEAR_GOLD: (255, 235, 156),
    MeepleColor.SOLID_BRONZE: BRONZE,
    MeepleColor.CLEAR_BRONZE: (226, 173, 125),
    MeepleColor.CLEAR_TAN: TAN,
    MeepleColor.EMPTY: GRAY, # Added
}

class Phase(Enum):
    EMPTY = -1
    START = 0
    PLACE_TOKEN = 1
    PLAY = 2
    END_ROUND = 3
    GAME_OVER = 9

    @classmethod
    def from_int(cls, value):
        for item in cls:
            if item.value == value:
                return item
        print(f"Warning: Unknown Phase value {value} received. Defaulting to EMPTY.")
        return cls.EMPTY

class TradePostType(Enum):
    NONE = 0
    POST = 1
    CENTER = 2

    @classmethod
    def from_int(cls, value):
        for item in cls:
            if item.value == value:
                return item
        return cls.NONE

CITY_DATA = [
    (1, "Agadez", "Tuareg", "Iron work", "Silver cross"),
    (2, "Bandiagara", "Dogon", "Onions/tobacco", "Dogon mask"),
    (3, "Dinguiraye", "Fulani", "Cattle", "Wedding blanket"),
    (4, "Dosso", "Songhai-Zarma", "Cotton", "Silver headdress"),
    (5, "Hemang", "Akan", "Kente cloth", "Gold weight"),
    (6, "Katsina", "Housa", "Kola nuts", "Holy book"),
    (7, "Linguère", "Wolof", "Casava/peanut", "Gold necklace"),
    (8, "Ouagadougou", "Dagbani-Mossi", "Horses", "Bronze bracelet"),
    (9, "Oudane", "Arab", "Camel", "Bronze incense burner"),
    (10, "Oyo", "Yoruba", "Ivory", "Ivory bracelet"),
    (11, "Ségou", "Mande/Bambara", "Millet", "Chiwara"),
    (12, "Sikasso", "Senoufo", "Brass jewelry", "Kora"),
    (13, "Tabou", "Kru", "Pepper", "Kru boat"),
    (14, "Warri", "Idjo", "Palm Oil", "Coral necklace"),
    (15, "Timbuktu", "Songhai", "Salt", "Gold crown")
]