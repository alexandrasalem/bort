from enum import Enum

MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"
PROD_DELIM_LEFT = " <"
PROD_DELIM_RIGHT = " >"


class LettersStrategy(Enum):
    BULLETS = "bullets"
    COMPACT_BULLETS = "compact_bullets"
    TRANSPARENT = "transparent"
    SPACES = "spaces"


class ExperimentalConfiguration(Enum):
    BOTH_FULL = "both_full"
    BOTH_WEIGHTED = "both_weighted"
    ONLY_PRONOUNCE = "only_pronounce"
    ONLY_SPELL = "only_spell"

    @property
    def _proportions(self):
        return {
            self.BOTH_FULL: (1.0, 1.0),
            self.BOTH_WEIGHTED: (0.5, 0.5),
            self.ONLY_PRONOUNCE: (1.0, 0.0),
            self.ONLY_SPELL: (0.0, 1.0),
        }[self]

    @property
    def pronounce_proportion(self):
        return self._proportions[0]

    @property
    def spell_proportion(self):
        return self._proportions[1]
