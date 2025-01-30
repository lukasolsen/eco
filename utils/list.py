from typing import List, TypeVar

T = TypeVar("T")

class LimitedList(List[T]):
    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size

    def append(self, item: T) -> None:
        """Override append to maintain a fixed size."""
        if len(self) >= self.max_size:
            self.pop(0)
        super().append(item)
