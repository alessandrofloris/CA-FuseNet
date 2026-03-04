class ITWPolimiClassMapping:
    def __init__(self):
        self._mapping = {
            0: "cleaning",
            1: "crouching",
            2: "jumping",
            3: "laying",
            4: "riding",
            5: "running",
            6: "scooter",
            7: "sitting",
            8: "sittingTogether",
            9: "sittingWhileCalling",
            10: "sittingWhileDrinking",
            11: "sittingWhileEating",
            12: "sittingWhileHoldingBabyInArms",
            13: "sittingWhileTalkingTogether",
            14: "sittingWhileWatchingPhone",
            15: "standing",
            16: "standingTogether",
            17: "standingWhileCalling",
            18: "standingWhileDrinking",
            19: "standingWhileEating",
            20: "standingWhileHoldingBabyInArms",
            21: "standingWhileHoldingCart",
            22: "standingWhileHoldingStroller",
            23: "standingWhileLookingAtShops",
            24: "standingWhileTalkingTogether",
            25: "standingWhileWatchingPhone",
            26: "walking",
            27: "walkingTogether",
            28: "walkingWhileCalling",
            29: "walkingWhileDrinking",
            30: "walkingWhileEating",
            31: "walkingWhileHoldingBabyInArms",
            32: "walkingWhileHoldingCart",
            33: "walkingWhileHoldingStroller",
            34: "walkingWhileLookingAtShops",
            35: "walkingWhileTalkingTogether",
            36: "walkingWhileWatchingPhone"
        }

    def get_mapping(self):
        """Returns the whole class mapping as a dictionary."""
        return self._mapping

    def get_class_name(self, index):
        """Returns the name of the class given its index."""
        return self._mapping.get(index, "Unknown")

    def get_id_by_name(self, name):
        """Returns the index of the class given its name."""
        inv_map = {v: k for k, v in self._mapping.items()}
        return inv_map.get(name, -1)

    def get_macro_categories(self):
        """
        Utility for grouping the composite classes into their macro-activities.
        """
        groups = {
            "sitting": [7, 8, 9, 10, 11, 12, 13, 14],
            "standing": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            "walking": [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
            "other": [0, 1, 2, 3, 4, 5, 6]
        }
        return groups