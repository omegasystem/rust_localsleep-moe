def is_cjk(ch: str) -> bool:
    """Checks if a character is within CJK Unified Ideographs or fullwidth punctuation range."""
    if not ch:
        return False
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF) or
        (0x3400 <= cp <= 0x4DBF) or
        (0x20000 <= cp <= 0x2A6DF) or
        (0x3000 <= cp <= 0x303F) or
        (0xFF00 <= cp <= 0xFFEF)
    )

class CJKStreamCleaner:
    """
    Eliminates weird SentencePiece spaces between consecutive CJK characters during streaming.
    Preserves single space between CJK and English/numbers (e.g. 'Kuro, Leon 的 AI Familiar').
    """
    def __init__(self):
        self.last_char = ""

    def clean(self, piece: str) -> str:
        if not piece:
            return ""

        # Case 1: If previous char was CJK and this token starts with space before another CJK char:
        if is_cjk(self.last_char) and piece.startswith(" ") and len(piece) > 1 and is_cjk(piece[1]):
            piece = piece[1:]
        elif is_cjk(self.last_char) and piece == " ":
            # Suppress bare space after CJK (wait for next token to confirm if it's CJK)
            piece = ""

        if piece:
            self.last_char = piece[-1]
        return piece
