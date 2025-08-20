from enum import Enum


class KeyCommand(Enum):
    """Enumeration of all available IR key commands"""
    
    # Navigation Keys
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    OK = "OK"
    SELECT = "SELECT"
    
    # Menu Keys
    HOME = "HOME"
    MENU = "MENU"
    BACK = "BACK"
    EXIT = "EXIT"
    GUIDE = "GUIDE"
    INFO = "INFO"
    
    # Playback Control
    PLAY = "PLAY"
    PAUSE = "PAUSE"
    STOP = "STOP"
    PLAY_PAUSE = "PLAY_PAUSE"
    FAST_FORWARD = "FAST_FORWARD"
    REWIND = "REWIND"
    SKIP_FORWARD = "SKIP_FORWARD"
    SKIP_BACKWARD = "SKIP_BACKWARD"
    RECORD = "RECORD"
    
    # Channel Control
    CHANNEL_UP = "CHANNEL_UP"
    CHANNEL_DOWN = "CHANNEL_DOWN"
    LAST_CHANNEL = "LAST_CHANNEL"
    
    # Volume Control
    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"
    MUTE = "MUTE"
    
    # Number Keys
    NUM_0 = "0"
    NUM_1 = "1"
    NUM_2 = "2"
    NUM_3 = "3"
    NUM_4 = "4"
    NUM_5 = "5"
    NUM_6 = "6"
    NUM_7 = "7"
    NUM_8 = "8"
    NUM_9 = "9"
    
    # Color Keys
    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    BLUE = "BLUE"
    
    # Special Keys
    POWER = "POWER"
    INPUT = "INPUT"
    SETTINGS = "SETTINGS"
    SEARCH = "SEARCH"
    FAVORITES = "FAVORITES"
    APPS = "APPS"
    NETFLIX = "NETFLIX"
    AMAZON = "AMAZON"
    YOUTUBE = "YOUTUBE"
    
    # Additional Control
    PAGE_UP = "PAGE_UP"
    PAGE_DOWN = "PAGE_DOWN"
    SUBTITLE = "SUBTITLE"
    AUDIO = "AUDIO"
    HELP = "HELP"
    
    @classmethod
    def from_string(cls, value: str) -> 'KeyCommand':
        """Get KeyCommand from string value"""
        value_upper = value.upper()
        for key in cls:
            if key.value == value_upper:
                return key
        raise ValueError(f"Unknown key command: {value}")
    
    @classmethod
    def get_navigation_keys(cls) -> list['KeyCommand']:
        """Get all navigation keys"""
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT, cls.OK, cls.BACK]
    
    @classmethod
    def get_number_keys(cls) -> list['KeyCommand']:
        """Get all number keys"""
        return [cls.NUM_0, cls.NUM_1, cls.NUM_2, cls.NUM_3, cls.NUM_4,
                cls.NUM_5, cls.NUM_6, cls.NUM_7, cls.NUM_8, cls.NUM_9]